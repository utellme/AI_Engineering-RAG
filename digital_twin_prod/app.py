import os
from openai import OpenAI
import gradio as gr
from pprint import pprint
import uuid
import chromadb
import random
import requests
import json
import re



#------------------------------------------------------
# Setup
#------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception ("API key is missing.")
client = OpenAI()

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"



#------------------------------------------------------
# Documents
#------------------------------------------------------
# Default text if fail to read from the file
document_overview = """
update the text here or provide a file 'document_overview' covering your overview
"""

document_professional_experiences = """
update the text here or provide a file 'document_professional_experiences' covering your professional experiences
"""
document_personal_experiences_details  = """
update the text here or provide a file 'document_personal_experiences_details' covering your details about your professional experiences sharing real world examples in details such as
your strenghts, your weaknesses, your leadership style, your career goals, your success criteria, your cross functional skills achievements, your most accomplished projects and so on...
"""

document_people_management_experience = """
update the text here or provide a file 'document_people_management_experience' covering your details about your people management experiences sharing real world examples in details such as
your team building, team support, team culture, team accountability, hiring strategy and so on ....
"""

document_project_retros = """
update the text here or provide a file 'document_project_retros' experiences covering your details about your project retros experiences sharing real world examples in details such as
your team success, project success, team failures, project failures, learning, retrospectives, outcomes, tradeoffs, challenges and so on ....
"""

document_behavioral_motivational = """
update the text here or provide a file 'document_behavioral_motivational' experiences covering your details about your behavioral and motivational experiences sharing real world examples in details such as
covering in detaila about Conflict, Disagreement, Accountability, Tough Call, Continuous growth, influence, effective cross collaboration and so on ....
"""

#------------------------------------------------------
# Read input documents for a file
#------------------------------------------------------
file_names = ["document_overview", "document_professional_experiences", "document_personal_experiences_details", "document_people_management_experience", "document_project_retros"]

for doc in file_names:
    filename = doc + ".txt"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            if(doc == "document_overview"):
                document_overview = file.read()
            if(doc == "document_professional_experiences"):
                document_professional_experiences = file.read()
            if(doc == "document_personal_experiences_details"):
                document_personal_experiences_details = file.read()
            if(doc == "document_personal_experiences_details"):
                document_people_management_experience = file.read()
            if(doc == "document_project_retros"):
                document_project_retros = file.read()
            if(doc == "document_behavioral_motivational"):
                document_behavioral_motivational = file.read()     
    except FileNotFoundError:
        # If the file doesn't exist, use the default text instead
        print(f"file \"{doc}\" not found. Will use default text\n")


#------------------------------------------------------
# Chunking Function
#------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long text into overlapping chunks.
 
    Rules:
      - Each chunk is at most `chunk_size` characters.
      - Consecutive chunks overlap by ~`overlap` characters.
      - Every chunk ends on a sentence boundary (., !, ?).
      - No chunk ever cuts mid-sentence.
 
    Args:
        text:       The input text to split.
        chunk_size: Maximum number of characters per chunk (default 500).
        overlap:    Approximate number of characters to overlap between
                    consecutive chunks (default 50).
 
    Returns:
        A list of chunk strings.
    """
    # Split into sentences, keeping the terminal punctuation attached.
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text.strip())
 
    # Remove empty entries that can arise from split
    sentences = [s.strip() for s in sentences if s.strip()]
 
    chunks: list[str] = []
    start_idx = 0          # index into `sentences` where the current chunk begins
 
    while start_idx < len(sentences):
        current_chars = 0
        end_idx = start_idx  # will advance as long as sentences fit
 
        # Greedily add sentences until adding the next one would exceed chunk_size
        while end_idx < len(sentences):
            candidate = sentences[end_idx]
            # Account for the space separator between sentences in the chunk
            separator_len = 1 if end_idx > start_idx else 0
            if current_chars + separator_len + len(candidate) > chunk_size:
                break
            current_chars += separator_len + len(candidate)
            end_idx += 1
 
        # If we couldn't fit even a single sentence, force-include it to avoid
        # an infinite loop (the sentence itself is longer than chunk_size).
        if end_idx == start_idx:
            end_idx = start_idx + 1
 
        chunk = " ".join(sentences[start_idx:end_idx])
        chunks.append(chunk)
 
        # ------------------------------------------------------------------ #
        # Determine the next start_idx so that the new chunk overlaps the     #
        # current one by ~`overlap` characters.                                #
        # Strategy: walk *backwards* from end_idx, summing sentence lengths,  #
        # until we have accumulated at least `overlap` characters.  The       #
        # sentence just before that threshold becomes the new start.           #
        # ------------------------------------------------------------------ #
        accumulated = 0
        new_start = end_idx  # fallback: no overlap (start fresh)
 
        for i in range(end_idx - 1, start_idx - 1, -1):
            accumulated += len(sentences[i]) + (1 if i < end_idx - 1 else 0)
            if accumulated >= overlap:
                new_start = i
                break
 
        # Guard against an infinite loop: always advance by at least one sentence
        if new_start <= start_idx:
            new_start = start_idx + 1
 
        start_idx = new_start
 
    return chunks
#------------------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB
#------------------------------------------------------

### create chunks
ids = []
chunks = []
metadatas = []

documents = [
    {"text": document_overview, "source": "Dipesh's overview"},
    {"text": document_professional_experiences, "source": "Dipesh's Professional Experiences"},
    {"text": document_personal_experiences_details, "source": "Dipesh's Experiences in Details"},
    {"text": document_people_management_experience, "source": "Dipesh's People Management Experiences"},
    {"text": document_project_retros, "source": "Dipesh's Project Retrospectives"},
    {"text": document_behavioral_motivational, "source": "Dipesh's Behavioral and Motivational Experiences"}
    
]

# prepare data for storage
for doc in documents:
    chunks_ = chunk_text(doc['text'], chunk_size=600, overlap=50)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source":doc['source'] , "chunk_index": i} for i in range(len(chunks_))]

    # add to the list
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

### embed chunks
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks

)
embeddings = [item.embedding for item in response.data]

# verify embeddings
print(f"Generated: {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

### store in ChromaDB vector

#initialize client using Peersistent storage
chroma_client = chromadb.PersistentClient(path="./twin_chroma_db_2")
collection = chroma_client.get_or_create_collection(name="digital_twin")

#empty the collectioon data before adding for testing purpose
if(collection.get()["ids"]):
    collection.delete(collection.get()["ids"])

# add data into collection
collection.add(
    ids= ids,
    embeddings=embeddings,
    documents = chunks,
    metadatas=metadatas
    )

#pprint(collection.get())

#------------------------------------------------------
# Tool calling # 1 (push over)
#------------------------------------------------------
def send_notification(message:str):
    # handling of the failure responsee
    if pushover_user is None or pushover_token is None:
        return "Notification failed: Pushover not configured"
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent successfully: {message}"


send_notification_function = {

    "name" : "send_notification",
    "description": "Sends a push notification to the real Dipesh via pushover. Use this when:\
        1) Someone wants to get in touch, hire or collaborate\
            - ask for their name and contact details first, then send notification to Dipes with the name and contact details.\
        2) YOu don't know the answer to a queesiton about Dipesh - send automatically without asking, include the question so h can add this info later",
    "parameters": {
        "type": "object",
        "properties": {

            "message": {
                "type": "string",
                "description": "The notification message to send to the user's device"
            }
        },
        "required":["message"]
    }
}

tools = [{"type": "function", "function": send_notification_function}]

#------------------------------------------------------
# Tool calling # 2 (roll dice)
#------------------------------------------------------
# add another notification - dice roll

#Simulates rolling  a single six-sided one
def dice_roll():
    result = random.randint(1,6)
    return result

#DESCRIBE FUNCTION FORM LLM
dice_roll_function = {

    "name" : "dice_roll",
    "description": "simulate rolling a dice to get a random number between 1 to 6. Use this when user wants to roll a dice and get a result",
    "parameters": {
        "type": "object",
        "properties": {},
        "required":[]
    }
}

# add function to teh list of tools of LLM
tools.append({"type": "function", "function": dice_roll_function})
#print(tools)

#------------------------------------------------------
# Tool calling function
#------------------------------------------------------
# handle tool call
def handle_tool_call(tool_calls):
    # .....
    # return what to add to our "coontext" about the tool call results, a dictionary

    tool_call_results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name

        if(function_name == "send_notification"):
            args = json.loads(tool_call.function.arguments)
            #send notification
            result = send_notification(args["message"])
            content = result
        elif (function_name == "dice_roll"):
            content = f"Dice Roll: {dice_roll()}"
        else:
            content = f"Unknown function: {function_name}"

        print(content)
        tool_call_result = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": content
            }
        tool_call_results.append(tool_call_result)

    return tool_call_results


#------------------------------------------------------
# System Message
#------------------------------------------------------
system_message = """
You are a digital twin of Dipesh Valia. When people talk to you, you respond AS Dipesh - in first pereson, using his voice, personality and knowledge.

Important: do not make things up. If you don't know an answer, say you don't know.
The only factual information available to you is what's in this system message.
You cannot get any more facts about DIpesh from the internet or make them up.

Important: Whenever you don't know something about Dipesh,, ALWAYS use the send_notificaiton tool to alert the real Dipesh - do this automatically without asking the user.
"""

#------------------------------------------------------
# Main Response Function
#------------------------------------------------------
def response_openai(message, history):

    # user input message embeds
    response_query = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )

    # got vectors from the embedded model based on the query message
    query_embeddings = response_query.data[0].embedding

    # search embeds in chromadb
    results = collection.query(
        query_embeddings=[query_embeddings], # Your vector list
        n_results=10
    )
  
    # stich retrieved chunks together to creat the context for the responose
    context = "\n----\n".join(results["documents"][0])
    #print(f" dynamic context: {context}")

    print("\n------------------------------------------------\n")
   

    system_message_dynamic = system_message + context
    print(f" User Message ---> {message}")
    print(f" Systeem Enhanced Messagee Message ---> {system_message_dynamic}")
    
    messages = [{"role": "system", "content": system_message_dynamic}] + history + [{"role": "user", "content": message}, ]

    #print(f"response_openai history: {history}")

    response = client.chat.completions.create(
        model= "gpt-4.1-mini",
        messages= messages,
        tools=tools
    )
    message = response.choices[0].message

    #check if model wants to call a tool
    while message.tool_calls:
    
        #pprint(message.tool_calls)
        
        tool_call_results = handle_tool_call(message.tool_calls) # send list of tool calls
        messages.append(message)
        # 'extend' is adding a list to an existing list as compared to 'append' you are adding an element to a list.
        # no need to loop each element to add to an existing list
        messages.extend(tool_call_results)
        #print(f"response_openai before 2nd LLM call: {message}")
        response2 = client.chat.completions.create(
           messages = messages,
           model = "gpt-4.1-mini",
           tools = tools
        )
     
        message = response2.choices[0].message
        # print(f"response_openai message after 2nd LLM call: {message}")
    
    return message.content

#------------------------------------------------------
# Launch Gradio
#------------------------------------------------------
# Custom CSS to apply a light blue background
css = """
    .gradio-container {
        background-color: #E3F2FD; 
        color: #0D47A1; /* Dark blue color */
    }

    /* Specifically targets titles, descriptions, and labels */
    .gradio-container h1, .gradio-container p, .gradio-container span {
        color: #0D47A1 !important;
    }

    /* Ensures the chatbot message text is also dark blue */
    .message-text {
        color: #0D47A1 !important;
    }

    /* Dark Mode overrides */
    @media (prefers-color-scheme: dark) {
        .gradio-container {
            background-color: #0D1B2A; /* Dark navy background */
            color: #E0E1DD; /* Light gray/off-white text */
        }
    
        /* Ensure title and labels also switch */
        .gradio-container h1, .gradio-container p, .gradio-container span {
            color: #E0E1DD !important;
        }

        /* Ensures the chatbot message text is also Light gray/off-white text */
        .message-text {
            color: #E0E1DD !important;
        }
    }
"""

with gr.Blocks(css=css) as digital_twin:
    gr.ChatInterface(
            fn=response_openai,
            title = "Dipesh's Digital Twin",
            chatbot=gr.Chatbot(avatar_images=(None, "Dipesh3.jpeg")),
            description= "Chat with a digital twin of Dipesh Valia to know his professional/personal background, leadership qualities, skills, AI experience or anything else. Can also contact Dipesh. Digit twin will notify real Dipesh with your message and details",
            examples=["What's your professional background?", "Whats your personal background?", "What's your AI experience?", "How can I contact you?"]
        )
digital_twin.launch(inbrowser=True)