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



# biobert embeddings of the research papers based on title+abstract



with open('/kaggle/input/biobertembeddings-datafile-biobertweights/embeddings_final.pickle', 'rb') as handle:

    temp_embeddings = pickle.load(handle)



# metadata about these research papers    



temp_title_abstract=pd.read_csv("/kaggle/input/biobertembeddings-datafile-biobertweights/title_abstract.csv")
# metadata on the whole corpus of research papers from the CORD challenge

metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
temp_embeddings=temp_embeddings.tolist()
embeddings=[]
# selecting only those research papers which have a body text present and the length of title+abstract is greater than 40





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

        if length_text>=40:

            title_abstract=title_abstract.append({'cord_uid':cid,'sha':sha,'title':title12,'abstract':abstract12},ignore_index=True)

            embeddings.append(temp_embeddings[i])
embeddings=np.array(embeddings)
title_abstract.shape
embeddings.shape
from bert_serving.client import BertClient

bc = BertClient()
query_subtask_mapping2=pd.DataFrame(columns=['Queries ', 'Subtask mapping ', 'Question form of queries '])
'''

The next three cells contain a list of queries, a list of subtasks and a list of questions

q1 is a list of queries generated from the subtasks

s1 is a list of subtasks corresponding to the queries

q2 is a list of questions corresponding to the queries



q2 has been generated for a question-answer system



more information regarding these can be found in the task1-results/archive (2)/Covid19_queries_questions_subtasks.xlsx file

'''
q1=['data standards and nomenclature',

'information sharing and inter-sectoral collaboration',

'governmental public health',

'communicating with high-risk populations',

'clarify community measures',

'equity considerations and problems of inequity',

'data-gathering with standardized nomenclature',

'planners, providers response', 

'barriers to information-sharing',

'mitigating barriers to information-sharing',

'coverage policies related to testing',

'coverage policies related to treatment',

'coverage policies related to care',

'Mitigating threats to incarcerated',

'assuring access to diagnosis treatment', 

'reach marginalized and disadvantaged populations',

'gaps and problems of inequity in the Nation’s public health capability, capacity',

'funding for all citizens', 

'Misunderstanding around containment and mitigation',

'Communication for potential risk of disease',

'Risk communication and guidelines',

'targeting at risk populations’ families',

'communicating with target elderly',

'communicating with health workers',

'communicating with target high-risk populations', 

'investments in baseline public health response infrastructure',

'Integration of public health surveillance systems',

'integration of federal and state',

'recruit and support local expertise', 

'coordinate public, private, commercial and non-profit and academic'

]
s1=['What has been published about data standards and nomenclature?',

'What has been published about information sharing and inter-sectoral collaboration?', 

'What has been published about governmental public health?', 

'What has been published about communicating with high-risk populations?',

'What has been published to clarify community measures?',

'What has been published about equity considerations and problems of inequity?',

'Methods for coordinating data-gathering with standardized nomenclature.',

'Sharing response information among planners, providers, and others.',

'Understanding and mitigating barriers to information-sharing.',

'Understanding and mitigating barriers to information-sharing.',

'Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care',

'Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care',

'Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care',

'Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.',

'Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.',

'Measures to reach marginalized and disadvantaged populations.',

'Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.',

'Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.',

'Misunderstanding around containment and mitigation.',

'Communication that indicates potential risk of disease to all population groups.',

'Communication that indicates potential risk of disease to all population groups.',

'Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).',

'Modes of communicating with target high-risk populations (elderly, health care workers).',

'Modes of communicating with target high-risk populations (elderly, health care workers).',

'Modes of communicating with target high-risk populations (elderly, health care workers).',

'Value of investments in baseline public health response infrastructure preparedness',

'Integration of federal/state/local public health surveillance systems.',

'Integration of federal/state/local public health surveillance systems.',

'How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).',

'How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).'

]
q2=['data standards and nomenclature',

'information sharing and inter-sectoral collaboration',

'governmental public health',

'communicating with high-risk populations',

'clarify community measures',

'equity considerations and problems of inequity',

'data-gathering with standardized nomenclature',

'planners, providers response',

'barriers to information-sharing',

'mitigating barriers to information-sharing',

'coverage policies related to testing',

'coverage policies related to treatment',

'coverage plocies related to care',

'Mitigating threats to incarcerated',

'assuring access to diagnosis treatment',

'reach marginalized and disadvantaged populations',

'gaps and problems of inequity in the Nation’s public health capability, capacity',

'funding for all citizens',

'What are misunderstanding around containment and mitigation',

'Communication for potential risk of disease',

'Risk communication and guidelines',

'targeting at risk populations’ families',

'communicating with target elderly',

'communicating with health workers',

'communicating with target high-risk populations',

'investments in baseline public health response infrastructure',

'Integration of public health surveillance systems',

'integration of federal and state',

'recruit and support local expertise',

'How to coordinate public, private, commercial and non-profit and academic?'

]
query_subtask_mapping2['Queries ']=q1

query_subtask_mapping2['Subtask mapping ']=s1

query_subtask_mapping2['Question form of queries ']=q2
query_subtask_mapping2
# subtask to cluster mapping



subtask_cluster_mapping=pd.read_excel("/kaggle/input/task1-results/archive (2)/Mapping_To_Clusters_Updated_08042020.xlsx",sheet_name="Query Matching")
# paper to cluster mapping



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
# biobert embeddings of the queries

query_embeddings=bc.encode(queries)
query_subtask_mapping2.columns=['Queries','Subtask mapping','Question form of queries','Clusters']
# return the top k research papers from both cluster and the whole corpus



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
new_frame.to_csv("task__results.csv",index=False)