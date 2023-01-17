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
from scipy.spatial.distance import cdist

import subprocess



import matplotlib.pyplot as plt

import pickle





!pip install tensorflow==1.15

# Install bert-as-service

!pip install bert-serving-server==1.10.0

!pip install bert-serving-client==1.10.0

!cp /kaggle/input/biobert-pretrained /kaggle/working -r

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.index /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.index

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.data-00000-of-00001 /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.data-00000-of-00001

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.meta /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.meta
%%time



# Start the BERT server

bert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=512 -max_batch_size=32 -num_worker=2'

process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)
import pickle

with open('/kaggle/input/biobertembeddings-datafile-biobertweights/embeddings_final.pickle', 'rb') as handle:

    temp_embeddings = pickle.load(handle)



temp_title_abstract=pd.read_csv("/kaggle/input/biobertembeddings-datafile-biobertweights/title_abstract.csv")
metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
print(metadata.columns)

print(temp_title_abstract.columns)
temp_embeddings=temp_embeddings.tolist()
embeddings=[]
def check_length(text,abstract):

    if pd.isna(text)==True:

        return len(abstract)

    else:

        val=text+" "+abstract

        return len(val)



title_abstract=pd.DataFrame(columns=['cord_uid','sha','title','abstract'])

for i,row in temp_title_abstract.iterrows():

    cid=temp_title_abstract.loc[i,'cord_uid']

    ind_list=metadata.index[metadata['cord_uid']==cid].tolist()

    if len(ind_list)==0:

        continue

    ind=ind_list[0]

    pdf=metadata.loc[ind,'has_pdf_parse']

    pmc=metadata.loc[ind,'has_pmc_xml_parse']

    if pdf==False and pmc==False:

        continue

    else:

        title12=temp_title_abstract.loc[i,'title']

        abstract12=temp_title_abstract.loc[i,'abstract']

        sha=temp_title_abstract.loc[i,'sha']

        length_text=check_length(title12,abstract12)

        if length_text>=20:

            title_abstract=title_abstract.append({'cord_uid':cid,'sha':sha,'title':title12,'abstract':abstract12},ignore_index=True)

            embeddings.append(temp_embeddings[i])
embeddings=np.array(embeddings)
title_abstract.shape
embeddings.shape
from bert_serving.client import BertClient

bc = BertClient()
query_subtask_mapping2=pd.DataFrame(columns=['Queries ', 'Subtask mapping ', 'Question form of queries '])
q1=['smoking risk factor', 

'potential smoking risk', 

'pulmonary disease risk',

'Pregnancy risk', 

'neonates risk', 

'high risk patients', 

'helath migration for control', 

'risk of fatality', 

'symptomatic hospitalized patients risk', 

'hospitalized patients risk', 

'severity of disease', 

'Suceptibilty by population',

'risk for population',

'transmission dynamics', 

'reproductive number virus',

'incubation period virus', 

'serial interval virus', 

'economic impact of virus',

'difference in economy',

'socio economic risk virus',

'behavioural factor economy virus',

'public health risk', 

'public health mitigation',

'public health mitigation measures',

'Co-infection risk', 

'pre infection risk', 

'other respiratory disease', 

'health mitigation measures',

'co - morbidities',

'disease make more transmissible',

'other infection makes virus more virulent',

'other infection makes virus more transmissible'

]
s1=['Smoking, pre-existing pulmonary disease',

'Smoking, pre-existing pulmonary disease',

'Smoking, pre-existing pulmonary disease',

'Neonates and pregnant women',

'Neonates and pregnant women',

'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

'Public health mitigation measures that could be effective for control',

'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

'Susceptibility of populations',

'Susceptibility of populations',

'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors',

'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors',

'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors',

'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors',

'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

'Public health mitigation measures that could be effective for control',

'Public health mitigation measures that could be effective for control',

'Public health mitigation measures that could be effective for control',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

'Public health mitigation measures that could be effective for control',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities'

]
q2=['What are smoking risk factor?',

'What are potential smoking risk?',

'What are pulmonary disease risk?',

'What are Pregnancy risk?',

'What are neonates risk?',

'What are high risk patients?',

'What is health migration for control?',

'What is risk of fatality?',

'What are symptomatic hospitalized patients risk?',

'What are hospitalized patients risk?',

'What is severity of disease?',

'What is suceptibilty by population?',

'What is risk for population?',

'What are transmission dynamics?',

'What is reproductive number of the virus?',

'What is the incubation period of the virus?',

'What is serial interval of the virus?',

'What is economic impact of virus?',

'What is difference in economy due to the virus?',

'What are scoio economic risk of the virus?',

'What are behavioural factor economy virus?',

'What is public health risk?',

'What is public health mitigation?',

'What are public health mitigation measures?',

'What is co-infection risk?',

'What are pre infection risks?',

'What are other respiratory disease risks?',

'What are health mitigation measures?',

'What are co - morbidities?',

'Does disease make virus more transmissible?',

'Does other infection makes virus more virulent?',

'Does other infection makes virus more transmissible?'

]
query_subtask_mapping2['Queries ']=q1

query_subtask_mapping2['Subtask mapping ']=s1

query_subtask_mapping2['Question form of queries ']=q2
query_subtask_mapping2
subtask_cluster_mapping=pd.read_excel("/kaggle/input/task1-results/archive (2)/Mapping_To_Clusters_Updated_08042020.xlsx",sheet_name="Query Matching")
paper_cluster_mapping=pd.read_excel("/kaggle/input/task1-results/archive (2)/Final_Clusters_Keywords_UID.xlsx")
import ast

query_subtask_mapping2['Clusters']=""

for i,row in query_subtask_mapping2.iterrows():

    subtask=query_subtask_mapping2.loc[i,'Subtask mapping ']

    clust_ind=[]

    sub_ind=subtask_cluster_mapping.index[subtask_cluster_mapping['subtasks']==subtask].tolist()

    if len(sub_ind)>0:

        ind=sub_ind[0]

        clust_ind=ast.literal_eval(subtask_cluster_mapping.loc[ind,'Important_Clusters'])

    query_subtask_mapping2.at[i,'Clusters']=clust_ind
query_subtask_mapping2
queries=list(query_subtask_mapping2['Queries '])
query_embeddings=bc.encode(queries)
query_embeddings.shape
query_subtask_mapping2.columns=['Queries','Subtask mapping','Question form of queries','Clusters']
import scipy.spatial

def get_top_results(query_embed,cluster_search_embedding,cluster_search_list_temp,k):

    closest_n = min(len(cluster_search_embedding),k)

    distances = scipy.spatial.distance.cdist([query_embed], cluster_search_embedding, "cosine")[0]

    results = zip(range(len(distances)), distances)

    results = sorted(results, key=lambda x: x[1])

    ret_dict={}

    for idx, distance in results[0:closest_n]:

        cid=cluster_search_list_temp[idx]

        val=1-distance

        ret_dict[cid]=val

    return ret_dict
new_frame=pd.DataFrame(columns=['Queries','Subtask mapping','Question form of queries','Clusters','cord_uid','title','abstract','similarity','cluster','total'])
for i,row in query_subtask_mapping2.iterrows():

    query=query_subtask_mapping2.loc[i,'Queries']

    subtask_mapping=query_subtask_mapping2.loc[i,'Subtask mapping']

    ques=query_subtask_mapping2.loc[i,'Question form of queries']

    query_embed=list(query_embeddings[i])

    clusters=query_subtask_mapping2.loc[i,'Clusters']

    total_search_dict={}

    cluster_search_list_temp=[]

    cluster_search_dict={}

    for j in clusters:

        paper_list=paper_cluster_mapping.index[paper_cluster_mapping['Cluster_Names']==j].tolist()

        if len(paper_list)>0:

            for k in paper_list:

                cid=""

                if pd.isna(paper_cluster_mapping.loc[k,'cord_uid'])==True:

                    title=paper_cluster_mapping.loc[k,'Title']

                    plist=title_abstract.index[title_abstract['title']==title].tolist()

                    if len(plist)>0:

                        p=plist[0]

                        cid=title_abstract.loc[p,'cord_uid']

                else:

                    tid=paper_cluster_mapping.loc[k,'cord_uid']

                    tlist=title_abstract.index[title_abstract['cord_uid']==tid].tolist()

                    if len(tlist)>0:

                        cid=tid

                if cid!="":

                    cluster_search_list_temp.append(cid)

    cluster_search_embedding=[]

    for j in cluster_search_list_temp:

        id1_list=title_abstract.index[title_abstract['cord_uid']==j].tolist()

        if len(id1_list)>0:

            id1=id1_list[0]

            emb=list(embeddings[id1])

            cluster_search_embedding.append(emb)

    if len(cluster_search_embedding)>0:

        returned_dict=get_top_results(query_embed,cluster_search_embedding,cluster_search_list_temp,30)

        for o in returned_dict:

            cluster_search_dict[o]=returned_dict[o]

    total_search_embedding=embeddings.tolist()

    total_search_list_temp=list(title_abstract['cord_uid'])

    total_search_dict=get_top_results(query_embed,total_search_embedding,total_search_list_temp,30)

    combined_list_cid=[]

    for t in cluster_search_dict:

        combined_list_cid.append(t)

    for t in total_search_dict:

        combined_list_cid.append(t)

    combined_list_cid=list(set(combined_list_cid))

    for t in combined_list_cid:

        flag=0

        flag1=0

        similar=0

        if t in cluster_search_dict:

            flag=1

            similar=cluster_search_dict[t]

        if t in total_search_dict:

            flag1=1

            similar=total_search_dict[t]

        id12=title_abstract.index[title_abstract['cord_uid']==t].tolist()[0]

        title2=title_abstract.loc[id12,'title']

        if pd.isna(title2)==True:

            title2=""

        abstract2=title_abstract.loc[id12,'abstract']

        new_frame=new_frame.append({'Queries':query,'Subtask mapping':subtask_mapping,'Question form of queries':ques,'Clusters':clusters,'cord_uid':t,'title':title2,'abstract':abstract2,'similarity':similar,'cluster':flag,'total':flag1},ignore_index=True)
new_frame.to_csv("task2_results.csv",index=False)