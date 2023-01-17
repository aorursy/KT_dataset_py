#import 
import pandas as pd
import numpy as np
import re
import json
import os
import pickle
from ipywidgets import interact_manual, widgets

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

%matplotlib inline
def load_obj(filename):
    with open(filename, "rb") as fp: # Unpickling
        b = pickle.load(fp)
        return b

#dataset creation
def load_filenames(path = "/kaggle/input/CORD-19-research-challenge/"):
    filenames = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "json":
                filenames.append(root + "/" + file)
    
    return filenames

def format_body(file, colm):
    if not colm in file.keys():
        return ""
    body_text = file[colm]
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    for section, text in texts:
        texts_di[section] += text
    body = ""
    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    return body


def generate_dataset():
    #load papers filenames
    all_files = load_filenames()
    data = []
    
    #load summaries 
    summaries = load_obj("/kaggle/input/biobertsum-summaries/summaries.pkl")
    for file in tqdm(all_files):
        file = json.load(open(file, 'rb'))
        if file['paper_id'] in summaries:
            summ = summaries[file['paper_id']][0]
            score = summaries[file['paper_id']][1]
        else:
            summ = []
            score = []
            
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_body(file, 'abstract'),
            format_body(file, 'body_text'),
            summ, 
            score
            
        ]
        data.append(features)
    col_names = ['paper_id', 'title', 'abstract', 'text', 'summaries', 'score']
    dataset = pd.DataFrame(data, columns=col_names)
    return dataset
dataset = generate_dataset()
dataset.head(2)
import torch
import os
path = "/kaggle/input/papers-for-biobert/testset/"
files = os.listdir(path)
papers = []
for f in files:
    papers += torch.load(path + f)
from ipywidgets import interact_manual, widgets
from nltk.tokenize import word_tokenize 

def clean_text(text):
    tokens = word_tokenize(text)

    start = -1
    for i, token in enumerate(tokens):
        if start<0:
            if not(token.isdigit()) and not (token in (string.punctuation)):
                start = i
    tokens = tokens[start:]
    sentence = " ".join(tokens)
    return sentence


@interact_manual
def search_articles(
    paper_id= "-"):
    
    if dataset['paper_id'].str.contains(paper_id).sum() == 0:
        print ("Searching Summaries for a valid random paper . . .", end = "\n\n")
        loc = np.random.choice(range (len(dataset)))
        
        while  dataset.iloc[loc]["summaries"] == []:
            loc = np.random.choice(range (len(dataset)))
            
        paper_id = dataset.iloc[loc]["paper_id"]
    
    row = dataset.loc[dataset['paper_id'] == paper_id]
    print ("Paper id: ", np.array(row["paper_id"])[0], end = "\n\n")
    print ("Title: ", np.array(row["title"])[0], end = "\n\n")
    
    abstract = np.array(row['abstract'])[0]
    if abstract == "":
        print ("This paper does not incude an abstract!", end="\n\n")
    else:
        print ("Original Abstract of the paper: ")
        print (abstract, end = "\n\n")

    print ("Summaries: ", end = "\n\n")
    summaries = np.array(row['summaries'])[0]
    scores = np.array(row['score'])[0]
    for i, sent in enumerate(summaries):
        print (str (i + 1) + ". " + clean_text(sent))
        print ("Sentence importance: ", scores[i], end = '\n\n')
%%capture
# BioBERT dependencies
import subprocess
# Tensorflow 2.0 didn't work with the pretrained BioBERT weights
!pip install tensorflow==1.15
# Install bert-as-service
!pip install bert-serving-server==1.10.0
!pip install bert-serving-client==1.10.0

# We need to rename some files to get them to work with the naming conventions expected by bert-serving-start
!cp /kaggle/input/biobert-pretrained /kaggle/working -r
%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.index /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.index
%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.data-00000-of-00001 /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.data-00000-of-00001
%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.meta /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.meta
%%time
#starting BioBert Service

bert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=None -max_batch_size=32 -num_worker=2'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)

# Start the BERT client. It takes about 10 seconds for the bert server to start, which delays the client
from bert_serving.client import BertClient

bc = BertClient()
#%%time
from scipy.spatial.distance import cosine
import pickle
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import string
import string 


with open('/kaggle/input/summary-embeddings/summary_embs_df.pkl','rb') as f:
    emb_df = pickle.load(f)


def cosine_distance(v1, v2):
    distance = 1 - cosine(v1, v2)
    return distance

def answer_query(query,num_summaries=5):
    ##Encode the query with biobert
    qemb = bc.encode([query])
    
    
    
    relevant_embeddings = emb_df

    ## Compute similarities with relevant embeddings and querry
    a = np.array([cosine_distance(qemb[0],relevant_embeddings['embedding'][i]) for i in range(relevant_embeddings.shape[0])])
    asort = np.argsort(a)
    
    ## Print everything
    print('')
    print('Generated summaries of '+str(num_summaries)+' most relevant papers for query:')
    print('"'+query+'"')
    for i in range(1,num_summaries):
        print('--------------------Paper: ', i, " --------------------")
        print("From paper with paperId : "+relevant_embeddings['paper_id'][asort[-i]])
        print("Important sentences : ")
        soum = relevant_embeddings['summary'][asort[-i]]
        sents = soum.split('.')      
        for s in sents:
            print (clean_text(s))    
            print('...')
        print('With average sentence importance score : '+str(relevant_embeddings['sum_score'][asort[-i]]))


#create interact manual
def_query = "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery."

@interact_manual
def search_articles(query=def_query, num_of_papers = "5"):
    answer_query(query, num_summaries= int (num_of_papers))


