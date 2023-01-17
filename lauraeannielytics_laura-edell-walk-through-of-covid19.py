# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input director
from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('BtN-goy9VOY',width=600, height=400)
laura_meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
laura_meta_data.head()
laura_meta_data.info()
laura_meta_data.describe(include='all')
laura_meta_data.drop_duplicates(subset=['title'], inplace=True)

laura_meta_data.info()
def doi_url(d):

    if d.startswith('http://'):

        return d

    elif d.startswith('doi.org'):

        return f'http://{d}'

    else:

        return f'http://doi.org/{d}'
laura_meta_data.doi = laura_meta_data.doi.fillna('').apply(doi_url)

laura_meta_data.head()
laura_meta_data.isna().sum()
import numpy as np

import pandas as pd

import os

import json

import glob

import sys

sys.path.insert(0, "../")
## CODE to CHECK JSON STRUCTURE

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/252878458973ebf8c4a149447b2887f0e553e7b5.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file['metadata']
# metadata

laura_meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')



laura_meta_data.drop_duplicates(subset=['sha'], inplace=True)



def doi_url(d):

    if d.startswith('http://'):

        return d

    elif d.startswith('doi.org'):

        return f'http://{d}'

    else:

        return f'http://doi.org/{d}'



laura_meta_data.doi = laura_meta_data.doi.fillna('').apply(doi_url)
Json = pd.read_csv("../input/create-corona-csv-file/kaggle_covid-19_open_csv_format.csv")

Json = Json.iloc[1:, 1:].reset_index(drop=True)



# merge frames

cols_to_use = laura_meta_data.columns.difference(Json.columns)

all_data = pd.merge(Json, laura_meta_data[cols_to_use], left_on='doc_id', right_on='sha', how='left')



all_data.title = all_data.title.astype(str) # change to string, there are also some numeric values



all_data.head(2)