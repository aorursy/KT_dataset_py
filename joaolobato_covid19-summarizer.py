# Import common libraries and connect with Kaggle environment

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import dataset direclty from Kaggle

import glob
import json
import pandas as pd
from tqdm import tqdm
root_path = '/kaggle/input/CORD-19-research-challenge/'
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)
# Create a dataframe 


metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
# Dropped columns with mixed dtype as they were not needed

meta_df.drop(columns=['who_covidence_id', 'arxiv_id'], inplace=True)
meta_df.head()
# Check the number of articles without an abstract

meta_df.abstract.isnull().sum()
# Check the total number of articles in dataframe

meta_df.abstract.value_counts().sum()
# Percentage of articles without an abstract

meta_df.abstract.isnull().sum()/ meta_df.abstract.value_counts().sum()
# Select only those without an abstract

abstracts = meta_df[meta_df.abstract.isnull()]
pd.options.display.max_colwidth = 2000
abstracts.head()
# Instal transformers

#!pip install -U sentence-transformers


# Import libraries
 

from transformers import pipeline
import requests
import pprint
import time
pp = pprint.PrettyPrinter(indent=14)
# Create an easy summarizer thanks to HuggingFace (https://huggingface.co/transformers/main_classes/pipelines.html#summarizationpipeline) the implementation could not be easier

summarizer =  pipeline("summarization")
# Here I used a text version of the json file:  0001418189999fea7f7cbe3e82703d71c85a6fe5.json

f = open('/content/Absence of surface expression of feline infectious peritonitis virus (FIPV).txt', 'r')
feline = f.read()
feline 
# Generate abstracts


abs_bart = summarizer(feline[:1022], min_length = 5, pad_to_max_length=True)
abs_t5 = summarizer_t5(feline[:511], min_length=5, max_length=40)
