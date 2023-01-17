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

        if length_text>=40:

            title_abstract=title_abstract.append({'cord_uid':cid,'sha':sha,'title':title12,'abstract':abstract12},ignore_index=True)

            embeddings.append(temp_embeddings[i])
embeddings=np.array(embeddings)
title_abstract.shape
embeddings.shape
from bert_serving.client import BertClient

bc = BertClient()
query_subtask_mapping2=pd.DataFrame(columns=['Queries ', 'Subtask mapping ', 'Question form of queries '])
q1=[ 'policy recommendations on mitigation measures',

 'mechanism for rapidly sharing that information regarding policy recommendations on mitigation measures',

 'determine asymptomatic disease',

 'early detection of disease',

 'existing diagnostic platforms',

 'existing surveillance platforms',

 'support and coordination of local expertise',

 'Recruitment of local expertise',

 'support and coordination legal ethical communications and operational issues',

 'leverage universities and private laboratories for testing purposes',

 'Guidelines about best practices to states', 

 'communications to public health officials and the public',

 'point-of-care test',

 'rapid bed-side tests',

 'tradeoffs between speed accessibility and accuracy',

 'design and execution of targeted surveillance experiments',

 'testing using PCR',

'Surveilance experiments collecting longitudinal samples', 

 'Separation of assay development issues from instruments',

 'role of private sector migrate assays  on development issue onto devices',

 'track the evolution of the virus',

 'viral load to detect the pathogen',

 'biological and environmental sampling',

 'diagnostics host response markers',

 'diagonosis to detect early disease',

 'predict severe disease progression',

 'best clinical practice and efficacy of therapeutic interventions',

 'Policies and protocols for screening and testing',

 'effects on supplies associated with mass testing',

 'Technology roadmap for diagnostics',

 'developing and scaling up new diagnostic tests',

 'crirical funding for diagnostics',

 'holistic approaches to COVID-19 and future diseases',

 'New platforms and technology to improve response times to diseases',

 'Coupling genomics and diagnostic testing on a large scale',

 'rapid sequencing and bioinformatics on genomes',

 'sequencing with advanced analytics for unknown pathogens',

 'distinguishing naturally-occurring pathogens from intentional',

 'Health surveillance of humans',

 'potential sources of future spillover',

 'exposure to pathogens from evolutionary hosts',

 'exposure to pathogens from transmission hosts',

 'exposure to pathogens from environmental demographic and occupational factors'

]
s1=['How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible.',

'How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible.',

'Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).',

'Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).',

'Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.',

'Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.',

'Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.',

'Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.',

'Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.',

'National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).',

'National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).',

'National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).',

'Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.',

'Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.',

'Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.',

'Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).',

'Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).',

'Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).',

'Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.',

'Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.',

'Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.',

'Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.',

'Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.',

'Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.',

'Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.',

'Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.',

'Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.',

'Policies and protocols for screening and testing.',

'Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.',

'Technology roadmap for diagnostics.',

'Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.',

'Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.',

'New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.',

'New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.',

'Coupling genomics and diagnostic testing on a large scale.',

'Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.',

'Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.',

'Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.',

'One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.',

'One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.',

'One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.',

'One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.',

'One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.'

]
q2=['What are policy recommendations on mitigation measures?',

'What is mechanism for rapidly sharing that information regarding policy recommendations on mitigation measures?',

'How to determine asymptomatic disease?',

'How to early detect disease?',

'How are existing diagnostic platforms?',

'How are existing surveillance platforms?',

'How is recruitment support and coordination of local expertise?',

'How is the public, private,non-profit and academic expertise and capacity?',

'What are support and coordination legal ethical communications and operational issues?',

'How to leverage universities and private laboratories for testing purposes?',

'What are communications to public health officials and the public?',

'What are communications to public health officials and the public?',

'What is point-of-care test?',

'What are rapid bed-side tests?',

'What are tradeoffs between speed accessibility and accuracy?',

'What are design and execution of targeted surveillance experiments?',

'How to do testing using PCR?',

'What is impact of ad hoc local interventions?',

'Separation of assay development issues from instruments?',

'How to migrate assays onto devices?',

'How to track the evolution of the virus?',

'What is viral load to detect the pathogen?',

'What are biological and environmental sampling?',

'What are diagnostics host response markers?',

'How to detect early disease?',

'How to predict severe disease progression?',

'What are best clinical practice and efficacy of therapeutic interventions?',

'What are Policies and protocols for screening and testing?',

'What are effects on supplies associated with mass testing swabs and reagents?',

'What is technology roadmap for diagnostics?',

'How to develope and scale up new diagnostic tests?',

'Who is funding for diagnostics?',

'What is the holistic approaches to COVID-19 and future diseases?',

'What are new platforms and technology to improve response times to diseases?',

'How to couple genomics and diagnostic testing on a large scale?',

'How to do rapid sequencing and bioinformatics on genomes?',

'Sequencing with advanced analytics for unknown pathogens?',

'How to distinguishing naturally-occurring pathogens from intentional?',

'How to do health surveillance of humans?',

'What are the potential sources of future spillover?',

'Exposure to pathogens from evolutionary hosts?',

'Exposure to pathogens from transmission hosts?',

'Exposure to pathogens from environmental demographic and occupational factors?'

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
new_frame.to_csv("task7_results.csv",index=False)