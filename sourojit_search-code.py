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
query_subtask_mapping=pd.read_excel("/kaggle/input/task1-results/archive (2)/Covid19_queries_questions_subtasks .xlsx",sheet_name="Sheet1")
query_subtask_mapping2=query_subtask_mapping[69:93].reset_index(drop=True)
list(query_subtask_mapping2.columns)
query_subtask_mapping2=query_subtask_mapping2[['Queries ', 'Subtask mapping ', 'Question form of queries ']]
query_subtask_mapping2.at[15,'Queries ']="livestock infection sustainability"

query_subtask_mapping2.at[15,'Question form of queries ']="What are livestock infection sustainability?"
query_subtask_mapping2.head(30)
queries=list(query_subtask_mapping2['Queries '])
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
%%time



# Start the BERT server

bert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=512 -max_batch_size=32 -num_worker=2'

process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)
from bert_serving.client import BertClient
bc = BertClient()
query_embeddings=bc.encode(queries)
query_embeddings.shape
subtask_cluster_mapping=pd.read_excel("/kaggle/input/task1-results/archive (2)/Mapping_To_Clusters_Updated_08042020.xlsx",sheet_name="Query Matching")

subtask_cluster_mapping_species=pd.read_excel("/kaggle/input/task1-results/archive (2)/Mapping_To_Clusters_Updated_08042020.xlsx",sheet_name="Species")
subtask_cluster_mapping.head()
subtask_cluster_mapping_species.head(10)
subtask_list=query_subtask_mapping2['Subtask mapping '].unique().tolist()
subtask_list
query_subtask_mapping2.head(30)
query_subtask_mapping2.at[8,'Subtask mapping ']='Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.'

query_subtask_mapping2.at[9,'Subtask mapping ']='Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.'

query_subtask_mapping2.at[10,'Subtask mapping ']='Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.'

query_subtask_mapping2.at[11,'Subtask mapping ']='Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.'
query_subtask_mapping2.at[13,'Subtask mapping ']='Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.'

query_subtask_mapping2.at[14,'Subtask mapping ']='Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.'
species_list=['Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.','Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.','Animal host(s) and any evidence of continued spill-over to humans','Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.']
query_subtask_mapping2.columns
subtask_cluster_mapping_species.columns
subtask_cluster_mapping.columns
import ast

query_subtask_mapping2['Clusters']=""

for i,row in query_subtask_mapping2.iterrows():

    subtask=query_subtask_mapping2.loc[i,'Subtask mapping ']

    clust_ind=[]

    if subtask in species_list:

        sub_ind=subtask_cluster_mapping_species.index[subtask_cluster_mapping_species['subtasks']==subtask].tolist()

        if len(sub_ind)>0:

            ind=sub_ind[0]

            clust_ind=ast.literal_eval(subtask_cluster_mapping_species.loc[ind,'Important_Clusters'])

    else:

        sub_ind=subtask_cluster_mapping.index[subtask_cluster_mapping['subtasks']==subtask].tolist()

        if len(sub_ind)>0:

            ind=sub_ind[0]

            clust_ind=ast.literal_eval(subtask_cluster_mapping.loc[ind,'Important_Clusters'])

    query_subtask_mapping2.at[i,'Clusters']=clust_ind

        
query_subtask_mapping2
paper_cluster_mapping=pd.read_excel("/kaggle/input/task1-results/archive (2)/Final_Clusters_Keywords_UID.xlsx")
paper_cluster_mapping.shape
paper_cluster_mapping.head()
query_subtask_mapping2.columns=['Queries','Subtask mapping','Question form of queries','Clusters']
query_subtask_mapping2.head()
import pickle

with open('/kaggle/input/biobertembeddings-datafile-biobertweights/embeddings_final.pickle', 'rb') as handle:

    embeddings = pickle.load(handle)



title_abstract=pd.read_csv("/kaggle/input/biobertembeddings-datafile-biobertweights/title_abstract.csv")
title_abstract.head()
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
new_frame.to_csv("task3_results.csv",index=False)