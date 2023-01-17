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
!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz




from typing import List, Dict, Iterable, Tuple



import os

import json



from tqdm import tqdm



import spacy

from spacy.tokens import Span

from scispacy.abbreviation import AbbreviationDetector

from scispacy.umls_linking import UmlsEntityLinker



import glob

import pickle

import spacy

from spacy import displacy

from spacy.matcher import Matcher

from collections import Counter

import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed

from IPython.display import Image

import cv2

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition

from sklearn import datasets

from sklearn.cluster import KMeans

import gc
def doi_to_url(doi):

    if isinstance(doi, float):

        return None

    elif doi.startswith('http'):

        return str(doi)

    elif doi.startswith('doi'):

        return 'https://' + str(doi)

    else:

        return 'https://doi.org/' + str(doi)
df_meta = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

df_meta['url'] = df_meta.doi.apply(doi_to_url)

df_meta.head(3)
data_path = '../input/CORD-19-research-challenge'

json_files = glob.glob(f'{data_path}/**/**/*.json', recursive=True)

len(json_files)
def to_covid_json(json_files):

    jsonl = []

    for file_name in tqdm(json_files):

        row = {"doc_id": None, "title": None, "abstract": None, "body": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']

            

            abstract_list = [abst['text'] for abst in data['abstract']]

            abstract = "\n".join(abstract_list)

            row['abstract'] = abstract



            # And lastly the body of the text. 

            body_list = [bt['text'] for bt in data['body_text']]

            body = "\n".join(body_list)

            row['body'] = body

            

        jsonl.append(row)

    

    return jsonl

    



def get_data():

    try:

        with open('df_cache.pickle', 'rb') as f:

            df = pickle.load(f)

    except FileNotFoundError:

        df = pd.DataFrame(to_covid_json(json_files))

        with open('df_cache.pickle', 'wb') as f:

            pickle.dump(df, f)

    return df



df = get_data()

print(df.shape)

df.head(3)
print("Documents where title is empty but abstract is present ",len(df[(df['title'] == '') &  (df['abstract']!= '')]))

print("Documents where title is present but abstract are absent", len(df[(df['title'] != '') &  (df['abstract']== '')]))
def fill_abstract_with_title(abstract , title):

    if abstract == '':

        return title

    else:

        return abstract



def fill_title_with_abstract(abstract, title):

    if title == '':

        return abstract.split('\n')[0]

    else:

        return title
df['abstract'] = df.apply(lambda x: fill_abstract_with_title(x['abstract'], x['title']), axis = 1)

df['title'] = df.apply(lambda x: fill_title_with_abstract(x['abstract'], x['title']), axis = 1)
## Checking Again

print("Documents where title is empty but abstract is present ",len(df[(df['title'] == '') &  (df['abstract']!= '')]))

print("Documents where title is present but abstract are absent", len(df[(df['title'] != '') &  (df['abstract']== '')]))
covid19_names = {

    'covid',

    'coronavirus',

    'corona virus',

    'corona',

    'new virus',

    'COVID19',

    'COVID-19',

    '2019-nCoV',

    '2019-nCoV.',

#     'novel coronavirus',  # too ambiguous, may mean SARS-CoV

    'coronavirus disease 2019',

    'Corona Virus Disease 2019',

    '2019-novel Coronavirus',

    'SARS-CoV-2',

}



def has_covid19(text):

    for name in covid19_names:

        if text and name.lower() in text.lower():

            return True

    return False
df['title_has_covid19'] = df.title.apply(has_covid19)

df['abstract_has_covid19'] = df.abstract.apply(has_covid19)

# df['body_has_covid19'] = df.body.apply(has_covid19)

df_covid19 = df[df.title_has_covid19 | df.abstract_has_covid19]

print(df_covid19.shape)
df_covid19.to_excel('Corona Virus Dataframe.xlsx', index = False)
df_covid19.head()
example_text = df_covid19.iloc[0,2]

print(example_text)
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz
import en_ner_bc5cdr_md
nlp = spacy.load('en_core_sci_sm')

#nlp = spacy.load('en_ner_bc5cdr_md')

nlp_ner = en_ner_bc5cdr_md.load()



doc = nlp_ner(example_text)

colors = {

    'CHEMICAL': 'lightpink',

    'DISEASE': 'lightorange',

}

displacy.render(doc, style='ent', options={

    'colors': colors

})
def get_named_entity_label(text):

    doc = nlp(text)

    named_entities = []

    for ent in doc.ents:

        named_entities.append((ent.text,ent.label_))

    return named_entities 
df_covid19['Named_Entities_Abstract'] = df_covid19['abstract'].apply(lambda x: get_named_entity_label(x))

df_covid19['Named_Entities_Title'] = df_covid19['title'].apply(lambda x: get_named_entity_label(x))
df_covid19.head()
def get_named_entity_df(df):

    try:

        with open('df_named_entity_cache.pickle', 'rb') as f:

            df_covid19 = pickle.load(f)

        return df_covid19

    except FileNotFoundError:

        #df_spacy = annotate_with_spacy(df)

        with open('df_named_entity_cache.pickle', 'wb') as f:

            pickle.dump(df, f)

    return None
get_named_entity_df(df_covid19)