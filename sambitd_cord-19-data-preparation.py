# Set your own project id here

PROJECT_ID = 'msds498'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path, PurePath

import pandas as pd

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

import re

from ipywidgets import interact

import ipywidgets as widgets

import pandas as pd

from IPython.display import display

import os

import json

import glob

from tqdm import tqdm



!pip install nltk

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 
# upload data and list contents

input_dir = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{input_dir}/metadata.csv'

metadata = pd.read_csv(metadata_path,

                               dtype={'pubmed_id': str,'title': str,'abstract': str})

metadata.head()
metadata.info()
metadata.isnull().sum()

# sha is the unique id for papers contained in pdf_json folder. There are over 10000 missing values. Likely because 

# they make be referencing to papers contained in pmc_json folder. pmcid is the unique identifier for papers kept in

# pmc_json folder.
# So how many papers in the metadataset have abstract.

len(metadata) - metadata.abstract.isnull().sum()
# Unique paper ids of research papers contained in pdf_json folder

metadata.sha.nunique()
# Title of research papers

metadata.title.nunique()
# Number of research papers in pdf json folder 

all_json_pdf = glob.glob(f'{input_dir}/**/pdf_json/*.json',recursive=True)

len(all_json_pdf)
# Number of research papers in pmc json folder 

all_json_pmc = glob.glob(f'{input_dir}/**/pmc_json/*.json',recursive=True)

len(all_json_pmc)
# Read a file from PMC_JSON FOLDER

#with open(all_json_pmc[0]) as file:

#    first_entry = json.load(file)

#    print(json.dumps(first_entry[:200],indent=4))

# Read a file from PDF_JSON FOLDER

#with open(all_json_pdf[0]) as file:

#    first_entry = json.load(file)

#    print(json.dumps(first_entry,indent=4))
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            self.abstract = []

            #Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json_pdf[0])

print(first_row)
dict_ = {'paper_id': [], 'abstract': [],'body_text': []}

for idx, entry in enumerate(all_json_pdf):

    if idx % (len(all_json_pdf) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json_pdf)}')

    

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)



    

df_json_pdf = pd.DataFrame(dict_, columns=['paper_id', 'abstract','body_text'])

df_json_pdf.head()
metadata.columns
df_json_pdf.columns
df = pd.merge(metadata,df_json_pdf,left_on='sha',right_on='paper_id',how='left').drop('paper_id',axis=1)
df.isnull().sum()
# Body text updated to df table from json

# Total research papers in json folder is around 80744

# 72000 from 80744 papers have body text from pdf_json folder

df.body_text.notnull().sum()
len(df)
df.columns
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.body_text = []

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: ... {self.body_text[:200]}...'

first_row = FileReader(all_json_pmc[0])

print(first_row)
dict_ = {'paper_id': [],'body_text': []}

for idx, entry in enumerate(all_json_pmc):

    if idx % (len(all_json_pmc) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json_pmc)}')

    

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['body_text'].append(content.body_text)



    

df_json_pmc = pd.DataFrame(dict_, columns=['paper_id','body_text'])

df_json_pmc.head()
# we do not see rows with empty body text. This is good.

df_json_pmc[df_json_pmc.body_text == '']
# All 58950 papers have body text

df_json_pmc.body_text.isnull().sum()
df_json_pdf.columns
len(df_json_pdf)
df_json_pmc.columns
len(df_json_pmc)
metadata.columns
len(metadata)
# Merge the original merged dataset with the df_json_pmc dataset

# left means left outer join

df = pd.merge(df,df_json_pmc,left_on='pmcid',right_on='paper_id',how='left').drop('paper_id',axis=1)

len(df)
# Body text updated to df table from json

# Total research papers in json folder is around 80744

# 72000 from 80744 papers have body text from pdf_json folder

df.body_text_x.notnull().sum()
# All 58950 papers have body text. Hence 

# 181778 - 58950 = 122828 papers do not have body_text_y

df.body_text_y.isnull().sum()
df.columns
#Lets compare abstracts from json folder and metadata

# abstract_x from metadata and abstract_y from pdf_json

df[df.abstract_x != df.abstract_y].shape
df[df.abstract_x != df.abstract_y][['abstract_x','abstract_y','url']].tail(20)
# check metadata abstract column to see if null values exist

df.abstract_x.isnull().sum(),(df.abstract_x == '').sum()
# Check pdf_json abstract to see if null values exist

df.abstract_y.isnull().sum(),(df.abstract_y == '').sum()
df.iloc[13:16,18:22]
# Convert all columns to string and then replace abstract_y values to test

#df = df.astype(str)

df["abstract_y"] = df["abstract_y"].astype(str) 

df['abstract_y'] = np.where(df['abstract_y'].map(len) > 50, df['abstract_y'], "na")
df[df['abstract_y'].apply(lambda x: len(str(x)) <= 10)]
# check metadata abstract column to see if null values exist

df.abstract_x.isnull().sum(),(df.abstract_x == '').sum()
# abstract_y values are all filled now. This is what we had expected after the "na" treatment

df.isnull().sum()
# Over 2000 rows where abstract_x value is null but abstract_y value has data

df.loc[df.abstract_x.isnull() & (df.abstract_y != 'na')]
# replace abstract_x (metadata column) with abstract_y (pdf_json) value where abstract_x is null

df.loc[df.abstract_x.isnull() & (df.abstract_y != 'na'),'abstract_x'] = df[df.abstract_x.isnull() & (df.abstract_y != 'na')].abstract_y
# Do we have any remaining null abstract values. Not anymore. This is good.

# The null values have reduced which is what we had expected.

df.abstract_x.isnull().sum()
# the remaining missing values are also empty in json folder

(df.abstract_x.isnull() & ((df.abstract_y != 'na') | (df.abstract_y != 'na'))).sum()
# Lets get rid of the pdf_json abstract column and rename the metadata abstract column

df.rename(columns = {'abstract_x' : 'abstract'}, inplace = True)

df.drop('abstract_y',axis=1,inplace = True)

df.columns
# This is expected because body text comes from pdf and pmc folders

(df.body_text_x != df.body_text_y).sum()
# check pdf_json body text to see if null values exist

# # 72000 from 80744 papers have body text from pdf_json folder

# 181778 - 72000 = 109778 records have null value of body_text_x

df.body_text_x.isnull().sum(),(df.body_text_y == '').sum()
# This is expected because there are only ~50000 papers in json_pmc

df.body_text_y.isnull().sum()
# body_text_x is pdf_json. body_text_y comes from pmc_json

# Where available we use the text from pmc file trusting the statement quality

df.body_text_x.isnull().sum(),(df.body_text_y.isnull()).sum()
df.shape
(df.body_text_x != df.body_text_y).sum()
# There are 7000 rows where body_text_x is null but body_text_y is not null

df.loc[df.body_text_x.isnull() & df.body_text_y.notnull()]
df.iloc[1337].body_text_x[:500]
df.iloc[1337].body_text_y[:500]
df.body_text_x.isnull().sum(),df.body_text_y.isnull().sum()
# We are trusting the text from pmc folder to be of higher quality as it contains full text. 

# Hence we will replace with body_text_x with body_text_y where body_text_y exists

df.loc[df.body_text_y.notnull(),'body_text_x'] = df.loc[df.body_text_y.notnull(), 'body_text_y']
# Lets get rid of the pdf_pmc body text column and rename the body text column

df.rename(columns = {'body_text_x' : 'body_text'}, inplace = True)

df.drop('body_text_y',axis=1,inplace = True)

df.columns
# Body text null values have now decreased.

df.body_text.isnull().sum()
df.isnull().sum()
df_processed = pd.DataFrame(df)

# Drop records where title is Null

df_processed = df_processed.dropna(axis=0,subset=['title'])

df_processed.drop(df_processed.columns[[0,1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18]],axis=1,inplace=True)

df_processed.columns
df_processed.isnull().sum()
%env JOBLIB_TEMP_FOLDER=/tmp

#df.to_csv('cord19_df_merged.csv',index=False)

df_processed.to_csv('cord19_processed.csv',index=False)

df_subset = df_processed.sample(frac = 0.05).reset_index(drop=True)

df_subset.to_csv('cord19_processed_subset.csv',index=False)