# import necessary modules

import numpy as np

import glob

import json

import pandas as pd

import pickle

import spacy

from spacy import displacy

from spacy.matcher import Matcher

from tqdm import tqdm

import os

from collections import Counter

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from ipywidgets import interact, interactive, fixed

from IPython.display import Image

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition

from sklearn import datasets

from sklearn.cluster import KMeans

import gc
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
root_path = '/kaggle/input/CORD-19-research-challenge/'

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json[0])

print(first_row)
def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data
dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    

    try:

        content = FileReader(entry)

    except Exception as e:

        continue  # invalid paper format, skip

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['abstract'].append(content.abstract)

    dict_['paper_id'].append(content.paper_id)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 100 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # if more than 2 authors, take them all with html tag breaks in between

            dict_['authors'].append(get_breaks('. '.join(authors), 40))

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

    

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    

    # add doi

    dict_['doi'].append(meta_data['doi'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
df_covid2 = df_covid.rename(columns={"body_text": "body", "paper_id": "doc_id"})

df_covid2.head()
# delete some columns

df_covid3 = df_covid2.drop(['doi', 'authors', 'journal', 'abstract_summary'], axis=1)

df_covid3.head()

df = df_covid3
# SMOKING

# smoking, vaping

smoke1 = df



# count how many times smoking & similar user defined words appeared in "body" column

smoke1['count_smoke'] = smoke1.body.str.count("Smok|smok|vaping")



# only take those rows where the count is 5 or more

smoke2 = smoke1.loc[smoke1['count_smoke'] > 4] 



# iterates through index of dataframe and breaks the body cells up into a list of lists

from nltk import tokenize

smoke3 = []                  

for ind in smoke2.index:

    smoke3.append(tokenize.sent_tokenize(smoke2.body[ind]))

    

# only include sentences that have a "summary" keyword and other user defined similar keywords

# use a for loop to iterate over the each list item and another for loop to iterate over all lists



substr_smoke = [

    'summary',

    'study indicates',

    'suggest',

    'relationship',

    'conclusion',

    'illustrates',

    'reveals',

    'denote',

    'results',

    'outcome',

    'evaluated'

]



list_orig = smoke3

keywords = substr_smoke

smoke4 = []



for l_inner in list_orig:

    l_out = []

    for item in l_inner:

        for word in keywords:

            if word in item:

                l_out.append(item)

    smoke4.append(l_out)



# only include sentences that have keyword and other user defined similar keywords

smoke_names = [

    'smok',

    'vaping'

]



list_orig = smoke4

keywords = smoke_names

smoke5 = []



for l_inner in list_orig:

    l_out = []

    for item in l_inner:

        for word in keywords:

            if word in item:

                l_out.append(item)

    smoke5.append(l_out)



# turns the list into a dataframe

smoke6 = pd.DataFrame(smoke5)



# join together all columns of each row that have sentences in them

smoke6['sum'] = smoke6.stack().groupby(level=0).agg(' '.join)



# reduce to just the 'sum' column and turn into dataframe and reset indexes so can be properly merged

smoke7 = smoke6.loc[:, 'sum']

smoke8 = pd.DataFrame(smoke7)



# reset index of original dataframe

smoke9 = smoke2.reset_index(drop=True)



# merge the 'sum' column to the original dataframe

smoke10 = pd.merge(smoke9, smoke8, left_index=True, right_index=True)



# drop unnecessary rows for final dataframe

smoke11 = smoke10.drop(['abstract', 'body'], axis=1)



# sort with most counts of keyword at the top and reset row index

smoke12 = smoke11.sort_values('count_smoke',ascending=False)

smoke13 = smoke12.reset_index(drop=True)



# reaarange columns

smoke13 = smoke13[['title', 'count_smoke', 'sum', 'doc_id']]



# print off answer

smoke13.style.set_properties(subset=['sum'], **{'width': '300px'})
smoke13.style.set_properties(subset=['sum'], **{'width': '300px'})