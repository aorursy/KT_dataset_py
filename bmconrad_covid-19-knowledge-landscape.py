import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
base_dir = "/kaggle/input/CORD-19-research-challenge/"

"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""

"""
with open("/kaggle/input/CORD-19-research-challenge/metadata.readme") as file:
    print("\n".join([i for i in file.readlines()]))
"""
with open("/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/26aec9a28a4345276498c14e302ead7d96c7feee.json") as file:
    doc1 = json.loads("\n".join([i for i in file.readlines()]))
print(doc1.keys())
print(doc1["body_text"][:-3], doc1["body_text"][-3:])
from collections import defaultdict
import re
data = defaultdict(list)

idx=0
paperid2title = {}
for dirname, _, filenames in os.walk(base_dir):
    print(dirname)
    for filename in filenames:
        if filename.endswith(".json"):
            with open(os.path.join(dirname, filename)) as file:
                doc = json.loads("\n".join([i for i in file.readlines()]))
                paperid = doc["paper_id"] #+ "_" + doc["metadata"]["title"]  
                paperid2title[paperid] = doc["metadata"]["title"]
                for chunk in doc["body_text"]:
                    data[paperid].append(chunk["text"])
                    #data[idx].append(chunk["text"])
                idx+=1
            
list(data.keys())[:10]
len(list(data.keys()))
questions = ["Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery",
             "Prevalence of asymptomatic shedding and transmission (e.g., particularly children).", 
             "Seasonality of transmission",
             "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)",
             "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood)",
             "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)",
             "Natural history of the virus and shedding of it from an infected person",
             "Implementation of diagnostics and products to improve clinical processes",
             "Disease models, including animal models for infection, disease and transmission",
             "Tools and studies to monitor phenotypic change and potential adaptation of the virus",
             "Immune response and immunity",
             "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",
             "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",
             "Role of the environment in transmission"]
questions
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

paragraph_list = []
count = 0
print("Accumulating data")
for k,v in data.items():
    if count % 1000 == 0:
        print(count)
    for p in v:
        paragraph_list.append(p)
    count += 1
print("Total paragraphs accumulated {}".format(len(paragraph_list)))


print("Starting vectorizer")
max_features = 10000
fit = TfidfVectorizer(max_features=max_features).fit(paragraph_list)
document_matrix = fit.transform(paragraph_list)
questions_matrix = fit.transform(questions)
vocab = fit.vocabulary_
print("Created vectorizer")

from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
print("Getting similarities")
C = cosine_similarity(questions_matrix, document_matrix)
N = 10
window_size = 5
percent = 0.01
for i, row in enumerate(C):
    print("*** Question ({}) {}\n".format(i, questions[i]))
    selections = row.argsort()[::-1][:N].tolist()
    initial_selections = [sent_tokenize(j) for idx in selections for j in paragraph_list[idx-window_size:idx+(window_size+1)]]
    row_selections = "\n".join([j for i in initial_selections for j in i])
    print("*** Summary About Question ({})".format(i))
    print(summarize(row_selections, ratio=percent))
    #print("*** Keywords Regarding Question ({})".format(i))
    #print(keywords(row_selections, split=True, lemmatize=True, ratio=0.1))
