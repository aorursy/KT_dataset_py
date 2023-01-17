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
selected_papers_task3=pd.read_csv('/kaggle/input/task3-results/task3_results.csv')

task3_summ=pd.read_excel('/kaggle/input/task1-results/task3_results_summary.xlsx')
task3_summ=task3_summ[task3_summ['Name']!='TITLE'].reset_index()
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
task3_summ_list=list(task3_summ['summary'])

query_list=selected_papers_task3['Queries'].unique().tolist()
task3_summ_list
summ_embedding=bc.encode(task3_summ_list)

query_embedding=bc.encode(query_list)
import scipy.spatial

new_frame=pd.DataFrame(columns=['query','cord_uid','summary','similarity'])



for k in range(len(query_list)):

    query=query_list[k]

    query_emb=list(query_embedding[k])

    cord_id_index=selected_papers_task3.index[selected_papers_task3['Queries']==query].tolist()

    cord_id_list=[]

    for j in cord_id_index:

        cid=selected_papers_task3.loc[j,'cord_uid']

        cord_id_list.append(cid)

    cord_id_list_summ=[]

    cord_id_list_emb=[]

    for j in cord_id_list:

        ind_list=task3_summ.index[task3_summ['cord_uid']==j].tolist()

        for p in ind_list:

            text=task3_summ.loc[p,'summary']

            cord_id_list_summ.append(text)

            summm_embed=list(summ_embedding[p])

            cord_id_list_emb.append(summm_embed)

    

    distances = scipy.spatial.distance.cdist([query_emb], cord_id_list_emb, "cosine")[0]

    results = zip(range(len(distances)), distances)

    results = sorted(results, key=lambda x: x[1])

    closest_n = min(len(cord_id_list_emb),5)       

    for idx, distance in results[0:closest_n]:

        text1=cord_id_list_summ[idx]

        cord_uid1=task3_summ.index[task3_summ['summary']==text1].tolist()[0]

        cid=task3_summ.loc[cord_uid1,'cord_uid']

        val=1-distance

        new_frame=new_frame.append({'query':query,'cord_uid':cid,'summary':text1,'similarity':val},ignore_index=True)

        

        

        

            

    

    

    
new_frame.to_csv("top_paragraph_per_query1234.csv",index=False)