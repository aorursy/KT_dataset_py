# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import json

from pprint import pprint

from copy import deepcopy

import numpy as np

import json

import glob

import sys



sys.path.insert(0, "../")



root_path = '/kaggle/input/'



corona_features = {"doc_id": [None], "source": [None], "title": [None],

                  "abstract": [None], "text_body": [None]}

corona_df = pd.DataFrame.from_dict(corona_features)



json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)
def return_corona_df(json_filenames, df):



    for file_name in json_filenames:



        row = {"doc_id": None, "source": None, "title": None,

              "abstract": None, "text_body": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            doc_id = data['paper_id']

            row['doc_id'] = doc_id

            row['title'] = data['metadata']['title']



            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 



            abstract_list = [abst['text'] for abst in data['abstract']]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract



            # And lastly the body of the text. 

            body_list = [bt['text'] for bt in data['body_text']]

            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            # Now just add to the dataframe. 

            

#             if source == 'b':

#                 row['source'] = "BIORXIV"

#             elif source == "c":

#                 row['source'] = "COMMON_USE_SUB"

#             elif source == "n":

#                 row['source'] = "NON_COMMON_USE"

#             elif source == "p":

#                 row['source'] = "PMC_CUSTOM_LICENSE"

            

            df = df.append(row, ignore_index=True)

    

    return df

    

corona_df = return_corona_df(json_filenames, corona_df)
corona_df.shape
corona_df.columns
risk_ind = []

count_risk = []



for i in corona_df['abstract']:

    if (str(i).lower().find('risk') != -1 and str(i).lower().find('covid') != -1):

    

        risk_ind.append(i)

        count_risk.append(i.lower().count('risk'))

corona_df_risk_covid = corona_df[corona_df['abstract'].isin(risk_ind)] 

corona_df_risk_covid['count_risk'] = count_risk
corona_df_risk_covid.drop_duplicates(['title'])
!pip install bert-extractive-summarizer

!pip install spacy

!pip install transformers==2.2.0
from summarizer import Summarizer



model = Summarizer()

summary = []

for i in corona_df_risk_covid['text_body']:

    result = model(i, min_length=60)

    full = ''.join(result)

    summary.append(full)
summary[0]