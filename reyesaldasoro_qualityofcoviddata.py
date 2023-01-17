import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json



# local path root_path = 'Desktop\COVID'



# path 

root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
!ls -la /kaggle/input/CORD-19-research-challenge/
meta_df.columns
meta_df.dtypes
num_empty_sha     = 0

num_empty_title   = 0

num_empty_doi     = 0

num_empty_authors = 0





empty_loc_sha     = pd.Series([0]);

empty_loc_title   = pd.Series([0]);

empty_loc_doi     = pd.Series([0]);

empty_loc_authors = pd.Series([0]);





for k in range(len(meta_df)):

    if isinstance(meta_df.sha[k], float):

        empty_loc_sha[num_empty_sha] = k

        num_empty_sha = num_empty_sha+1

    if isinstance(meta_df.title[k], float):

        empty_loc_title[num_empty_title] = k

        num_empty_title = num_empty_title+1

    if isinstance(meta_df.doi[k], float):

        empty_loc_doi[num_empty_doi] = k

        num_empty_doi = num_empty_doi+1

    if isinstance(meta_df.authors[k], float):

        empty_loc_authors[num_empty_authors] = k

        num_empty_authors = num_empty_authors+1 
print([num_empty_sha, num_empty_title, num_empty_doi, num_empty_authors])
meta_df.loc[empty_loc_sha].head()
meta_df.loc[empty_loc_title].head()
meta_df.loc[empty_loc_doi].head()
meta_df.loc[empty_loc_authors].head()
example_location   = 28804

meta_df.loc[example_location].title
meta_df.loc[example_location].authors
meta_df.loc[example_location].sha
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)

all_json[0]
for k in range(len(all_json)):

    if meta_df.loc[example_location].sha in all_json[k]:

        loc_example = k

        print(k)
all_json[loc_example]
with open(all_json[loc_example]) as file:

    content = json.load(file)
content['metadata']
meta_df.loc[8201].title
for k in range(len(all_json)):

    if meta_df.loc[8201].sha in all_json[k]:

        loc_example = k

        print(k)

with open(all_json[loc_example]) as file:

    content = json.load(file) 

    

content['metadata']
meta_df.loc[23473].title
for k in range(len(all_json)):

    if  meta_df.loc[23473].sha in all_json[k]:

        loc_example = k

        print(k)

with open(all_json[loc_example]) as file:

    content = json.load(file) 

    

content['metadata']['title']  
meta_df.loc[3895].title