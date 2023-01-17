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
data=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
data.shape
data.head()
cord_uid_list=data['cord_uid'].unique().tolist()
data_me=pd.read_csv("/kaggle/input/biobert-covid-embeddings/title_abstract.csv")
import pickle

with open('/kaggle/input/biobert-covid-embeddings/embeddings.pickle', 'rb') as handle:

    data_me_embedding=pickle.load(handle)
data_me_embedding.shape
data_me.head()
cord_prev_uid=data_me['cord_uid'].unique().tolist()
meta_df_title_abstract=data[pd.isna(data['abstract'])==False].reset_index(drop=True)
meta_df_title_abstract.shape
meta_df_title_abstract=meta_df_title_abstract[['cord_uid','sha','title','abstract']]
new_data=pd.DataFrame(columns=['cord_uid','sha','title','abstract'])
for i,row in meta_df_title_abstract.iterrows():

    cord_uid=meta_df_title_abstract.loc[i,'cord_uid']

    if cord_uid not in cord_prev_uid:

        sha=meta_df_title_abstract.loc[i,'sha']

        title=meta_df_title_abstract.loc[i,'title']

        abstract=meta_df_title_abstract.loc[i,'abstract']

        new_data=new_data.append({'cord_uid':cord_uid,'sha':sha,'title':title,'abstract':abstract},ignore_index=True)
new_data.shape
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

    

new_data['abstract_preprocessed']=new_data['abstract'].apply(lambda x:preprocess1(x))
def concat_text(a,b):

    if pd.isna(a)==True:

        return b

    return a+". "+b

new_data['text']=new_data.apply(lambda x:concat_text(x['title'],x['abstract_preprocessed']),axis=1)
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
new_data['text_preprocessed']=new_data['text'].apply(lambda x:preprocess_sentence(x))
data_list1=list(new_data['text_preprocessed'])
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
from bert_serving.client import BertClient
bc = BertClient()
embeddings=bc.encode(data_list1)

embeddings2=np.concatenate((data_me_embedding, embeddings), axis=0)
new_data=new_data[['cord_uid','sha','title','abstract']]

data_me=data_me[['cord_uid','sha','title','abstract']]

data_me=data_me.append(new_data, ignore_index=True)
data_me.shape
data_me.to_csv("/kaggle/working/title_abstract.csv",index=False)
with open('/kaggle/working/embeddings_final.pickle', 'wb') as handle:

    pickle.dump(embeddings2, handle)