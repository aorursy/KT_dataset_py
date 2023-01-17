# Uncomment and run this cell if you're on colab or kaggle.

!pip install scispacy scipy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz

!pip install tqdm -U
import pandas as pd 

import os

import numpy as np

import scispacy

import json

import spacy

from tqdm.notebook import tqdm

from scipy.spatial import distance

import ipywidgets as widgets
# Walks all subdirectories in a directory, and their files. 

# Opens all json files we deem relevant, and append them to

# a list that can be used as the "data" argument in a call to 

# pd.DataFrame.

def gather_jsons(dirName):

    

    # Get the list of all files in directory tree at given path

    # include only json with encoded id (40-character SHA hash)

    # Only length of filename is checked, but this should be sufficient

    # given the task.

    

    listOfFiles = list()

    for (dirpath, dirnames, filenames) in os.walk(dirName):

        listOfFiles += [os.path.join(dirpath, file) for file in filenames

                        if file.endswith("json")

                        and len(file) == 45]

    jsons = []

    

    print(str(len(listOfFiles)) + " jsons found! Attempting to gather.")

    

    for file in tqdm(listOfFiles):

        with open(file) as json_file:

            jsons.append(json.load(json_file))

    return jsons

        

        

# Returns a dictionary object that's easy to parse in pandas.

def extract_from_json(json):

    

    # For text mining purposes, we're only interested in 4 columns:

    # abstract, paper_id (for ease of indexing), title, and body text.

    # In this particular dataset, some abstracts have multiple sections,

    # with ["abstract"][1] or later representing keywords or extra info. 

    # We only want to keep [0]["text"] in these cases. 

    if len(json["abstract"]) > 0:

        json_dict = {

            "_id": json["paper_id"],

            "title": json["metadata"]["title"],

            "abstract": json["abstract"][0]["text"],

            "text": " ".join([i["text"] for i in json["body_text"]])

        }

        

    # Else, ["abstract"] isn't a list and we can just grab the full text.

    else:

        json_dict = {

            "_id": json["paper_id"],

            "title": json["metadata"]["title"],

            "abstract": json["abstract"],

            "text": " ".join([i["text"] for i in json["body_text"]])

        }



    return json_dict



# Combines gather_jsons and extract_from_json to create a

# pandas DataFrame object.

def gather_data(dirName):

    

    return(pd.DataFrame(data=[extract_from_json(json) for json in gather_jsons(dirName)]))
import os

PATH = "../input/CORD-19-research-challenge/"
# If in Colab, uncomment this code: 

# !unzip "/content/drive/My Drive/covid_data/CORD-19-research-challenge.zip"

df = gather_data(PATH)

df.to_csv("covid_data_full_20_03.csv", index=False)
# Uncomment if data already processed

df = pd.read_csv("covid_data_full_20_03.csv")
nlp = spacy.load("en_core_sci_lg")

# nlp = spacy.load("/users/platon/Downloads/en_core_sci_lg-0.2.4/en_core_sci_lg/en_core_sci_lg-0.2.4")
nlp.max_length=2000000
vector_list = []

for i in tqdm(df.index):

    doc = nlp(df.iloc[i].text)

    vector_list.append(

    {

    "_id": df.iloc[i]._id, 

    "vector": doc.vector,

    })

vector_df = pd.DataFrame(data=vector_list)
vector_df.to_csv("covid_vectors_20_03.csv",index=False)
with open('../input/covid-task-file/tasks.json', 'r') as f:

    tasks = json.load(f)
tasks_vector_list = []

for i in tqdm(range(len(tasks))):

    task = tasks[i]

    doc = nlp(task["description"])

    vec = doc.vector

    tasks_vector_list.append({"_id": f"task_{i}", "title": task["title"], "vector": vec})

    

tasks_vector_df = pd.DataFrame(data=tasks_vector_list)

tasks_vector_df.to_csv("tasks_vecs.csv",index=False)
distances = distance.cdist([value for value in tasks_vector_df["vector"]], [value for value in vector_df["vector"].values], "cosine")
w2v_searchable_df = vector_df.drop(columns=["vector"])
# Create a column with cosine distances for each query vs the sentence

for i in range(len(tasks)):

    w2v_searchable_df[f"task_{i}_distance"] = distances[i]

w2v_searchable_df.to_csv("covid_w2v_searchable_20_03.csv", index=False)
import json

with open('../input/covid-task-file/tasks.json', 'r') as f:

    tasks = json.load(f)

df = pd.read_csv("covid_data_full_20_03.csv")

w2v_searchable_df = pd.read_csv("covid_w2v_searchable_20_03.csv")

queries_df = pd.read_csv("tasks_vecs.csv")

vector_df = pd.read_csv("covid_vectors_20_03.csv")
for i in range(len(tasks)):

    columnName = f"task_{i}_distance"

    context = w2v_searchable_df.sort_values(by=columnName)[["_id"]][:10]

    ix = context["_id"].to_list()

    print(tasks[i]["title"] + "\n")

    for j in range(len(context.index)):

        print(f"Rank {j+1}: \nPaper ID: {ix[j]} \n" + str(df[df["_id"] == ix[j]].iloc[0]["text"])[:500] + "\n")