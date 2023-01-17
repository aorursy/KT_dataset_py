# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
task1=pd.read_excel("/kaggle/input/task1-results/31_3_task1_results.xlsx")
metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
biorxiv_clean=pd.read_csv("/kaggle/input/preprocessed-covid-data/biorxiv_clean.csv")
clean_comm_use=pd.read_csv("/kaggle/input/preprocessed-covid-data/clean_comm_use.csv")
clean_noncomm_use=pd.read_csv("/kaggle/input/preprocessed-covid-data/clean_noncomm_use.csv")
clean_pmc=pd.read_csv("/kaggle/input/preprocessed-covid-data/clean_pmc.csv")
queries=['incubation period','incubation period humans','incubation period range','incubation period human range','transmission range','transmission humans','transmission children','transmission seasonality','coronavirus survival','animal model of infection','immunity humans','immune response','immunity by age','environment ideal for spread','coronavirus environment','environmental transmission','transmission PPE','personal protective equipment transmission','movement control strategy','disease model','surface persistence','material persistence','environmental survival']
def preprocess1(x):

    x_list=x.split(" ")

    x_final=[]

    if x_list[0]=='Abstract' or x_list[0]=='Background':

        val=len(x_list)

        x_final=x_list[1:val]

    else:

        x_final=x_list

    sent=" ".join(x_final)

    return sent
query_dict={}

for j in queries:

    query_dict[j]=[]

    for i,row in task1.iterrows():

        que=task1.loc[i,'query']

        if que==j:

            title=task1.loc[i,'title']

            ind_list1=biorxiv_clean.index[biorxiv_clean['title']==title].tolist()

            ind_list2=clean_comm_use.index[clean_comm_use['title']==title].tolist()

            ind_list3=clean_noncomm_use.index[clean_noncomm_use['title']==title].tolist()

            ind_list4=clean_pmc.index[clean_pmc['title']==title].tolist()

            if len(ind_list1)>0:

                id1=ind_list1[0]

                body_text=biorxiv_clean.loc[id1,'text']

                body_text_list=body_text.split("\n\n")

                body_text_filt=[]

                for p in range(len(body_text_list)):

                    if p%2==1:

                        body_text_filt.append(body_text_list[p])

                for p in body_text_filt:

                    query_dict[que].append((i,p))

            elif len(ind_list2)>0:

                id1=ind_list2[0]

                body_text=clean_comm_use.loc[id1,'text']

                body_text_list=body_text.split("\n\n")

                body_text_filt=[]

                for p in range(len(body_text_list)):

                    if p%2==1:

                        body_text_filt.append(body_text_list[p])

                for p in body_text_filt:

                    query_dict[que].append((i,p))

            elif len(ind_list3)>0:

                id1=ind_list3[0]

                body_text=clean_noncomm_use.loc[id1,'text']

                body_text_list=body_text.split("\n\n")

                body_text_filt=[]

                for p in range(len(body_text_list)):

                    if p%2==1:

                        body_text_filt.append(body_text_list[p])

                for p in body_text_filt:

                    query_dict[que].append((i,p))

            elif len(ind_list4)>0:

                id1=ind_list4[0]

                body_text=clean_pmc.loc[id1,'text']

                body_text_list=body_text.split("\n\n")

                body_text_filt=[]

                for p in range(len(body_text_list)):

                    if p%2==1:

                        body_text_filt.append(body_text_list[p])

                for p in body_text_filt:

                    query_dict[que].append((i,p))

            else:

                abstract=task1.loc[i,'abstract']

                abstract=preprocess1(abstract)

                query_dict[que].append((i,abstract))

                

                

    
from scipy.spatial.distance import cdist

import subprocess



import matplotlib.pyplot as plt

import pickle as pkl





!pip install tensorflow==1.15

# Install bert-as-service

!pip install bert-serving-server==1.10.0

!pip install bert-serving-client==1.10.0

!cp /kaggle/input/biobert-pretrained /kaggle/working -r

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.index /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.index

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.data-00000-of-00001 /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.data-00000-of-00001

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.meta /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.meta

from nltk import word_tokenize

from nltk.corpus import stopwords

from string import punctuation

from scipy.spatial import distance

import nltk

nltk.download('stopwords')

nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_sentence(text):

    text = text.replace('/', ' / ')

    text = text.replace('.-', ' .- ')

    text = text.replace('.', ' . ')

    text = text.replace('\'', ' \' ')

    text = text.lower()



    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]



    return ' '.join(tokens)
%%time



# Start the BERT server

bert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=512 -max_batch_size=32 -num_worker=2'

process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)
from bert_serving.client import BertClient

bc = BertClient()
import scipy.spatial
new_frame=pd.DataFrame(columns=['query','pid','title','abstract','answer','similarity'])

closest_n = 5

for k in query_dict:

    quer=k

    quer=preprocess_sentence(quer)

    list_data=query_dict[k]

    query_list1=[quer]

    data_list1=[]

    pid_mapping={}

    cnt=0

    for j in list_data:

        a=preprocess_sentence(j[1])

        data_list1.append(a)

        pid_mapping[cnt]=j[0]

        cnt=cnt+1

    query_embedding=bc.encode(query_list1)

    embs=bc.encode(data_list1)

    new_embedding=embs.tolist()

    query_embed=query_embedding.tolist()

    

    distances = scipy.spatial.distance.cdist(query_embed, new_embedding, "cosine")[0]

    results = zip(range(len(distances)), distances)

    results = sorted(results, key=lambda x: x[1])

    for idx, distance in results[0:closest_n]:

        orig_id=pid_mapping[idx]

        title=task1.loc[orig_id,'title']

        pid=task1.loc[orig_id,'pid']

        abstract=task1.loc[orig_id,'abstract']

        answer=query_dict[k][idx][1]

        query1=k

        similarity=1-distance

        new_frame=new_frame.append({'query':k,'pid':pid,'title':title,'abstract':abstract,'answer':answer,'similarity':similarity},ignore_index=True)
new_frame.to_csv("ques_answer.csv",index=False)