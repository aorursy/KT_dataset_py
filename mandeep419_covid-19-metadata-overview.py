# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os 

import glob

import seaborn as sns

import sys
meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
meta_data.head(10)
meta_data.info()
meta_data.describe(include='all')
meta_data.drop_duplicates(subset=['title'], inplace= True)

meta_data.drop_duplicates(subset=['sha'], inplace=True)

meta_data.info()
def doi_url(d):

    if d.startswith('https://'):

        return d

    elif d.startswith('doi.org'):

        return f'https://{d}'

    else:

        return f'https://doi.org/{d}'

        
meta_data.doi= meta_data.doi.fillna('').apply(doi_url)

meta_data.head()
meta_data.isna().sum()
import json

file_path = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/019760388ec5fa9151add6cfe32178deda5433eb.json'

with open(file_path) as json_file:

    json_file = json.load(json_file)

json_file['metadata']
# Blank dataFrame to hold the metadata info



corona_features = {"doc_id" :[None], "source":[None], "title":[None], "abstract":[None], "text_body":[None]}

corona_df = pd.DataFrame.from_dict(corona_features)
corona_df
# Lets Grab all the json files 

root_path = '/kaggle/input/CORD-19-research-challenge'

json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)
# Now we just iterate over the files and populate the data frame. 



def return_corona_df(json_filenames, df, source):

    for file_name in json_filenames:

        row = {"doc_id":None, "source":None, "title":None, "abstract":None,"text_body":None}

        with open(file_name) as json_data:

            data = json.load(json_data)

            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']

        

            abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract

            body_list = []

            for _ in range(len(data['body_text'])):

                try:

                    body_list.append(data['body_text'][_]['text'])

                except:

                    pass



            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            # Now just add to the dataframe. 

            

            if source == 'b':

                row['source'] = "biorxiv_medrxiv"

            elif source == "c":

                row['source'] = "common_use_sub"

            elif source == "n":

                row['source'] = "non_common_use"

            elif source == "p":

                row['source'] = "pmc_custom_license"

            

            df = df.append(row, ignore_index=True)

    

    return df
corona_df = return_corona_df(json_filenames, corona_df, 'b')
corona_df.shape
corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')
# metadata 

# I added previous kernal output as a input data for this functions which is under covid-19-metadata-overview Folder

Json = pd.read_csv('/kaggle/input/covid-19-metadata-overview/kaggle_covid-19_open_csv_format.csv')
Json = Json.iloc[1:, 1:].reset_index(drop=True)



# merge frames

cols_to_use = meta_data.columns.difference(Json.columns)

all_data = pd.merge(Json, meta_data[cols_to_use], left_on='doc_id', right_on='sha', how='left')



all_data.title = all_data.title.astype(str) # change to string, there are also some numeric values



all_data.head(2)
# Check if the URLS are giving 200 status

import requests

import requests

headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.107 Safari/537.36' }

def doi_url_check(d):

    for url in d:

        code = requests.get(url, headers=headers).status_code

        if code == 200:

            print('ok') # or do something

        else:

            print(url) # not found
doi_url_check(meta_data.doi)