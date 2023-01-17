!pip install glove_python
# Based on https://www.kaggle.com/eliseygusev/perfect-lb-score-in-5-lines-of-code-1

import pandas as pd

import re

import string

import numpy as np



from glove import Corpus, Glove

from nltk.corpus import gutenberg

from multiprocessing import Pool

from scipy import spatial



import os

print(os.listdir("../input"))
#data = pd.read_csv('gap-coreference/gap-development.tsv', dtype={'A-coref': int, 'B-coref': int}, 

#                      delimiter='\t',encoding='utf-8')

data = pd.read_csv('https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv', dtype={'A-coref': int, 'B-coref': int}, 

                      delimiter='\t',encoding='utf-8')

data.head(2)
data.rename(columns={'A': 'A-Name', 'B': 'B-Name', 'A-coref': 'A', 'B-coref': 'B'}, inplace=True)

data['NEITHER'] = data.eval('1 - A - B')

data[['ID', 'Text', 'A', 'B', 'NEITHER']].to_csv("gap-development.csv", index=False)
train_data = pd.read_csv("gap-development.csv")

train_data.head(7)
train_data.Text[0]
def clean_text(text):

    text = text.lower()

    text = text.replace('[^\w\s]','')

    text = re.sub(r"\'s", " ", text)

    text = text.strip(' ')

    for elt in string.punctuation:

        text = text.replace(elt,'')

    text = re.sub(r"\'s", " ", text)

    return text
cleaned_train_text = []

for i in range(0,len(train_data)):

    cleaned_text = clean_text(train_data['Text'][i])

    cleaned_train_text.append(cleaned_text)

train_data['Text'] = pd.Series(cleaned_train_text).astype(str)
train_data.head()
cleaned_train_text[0]
train_data.to_csv("gap-development.csv", index=False)
Sentences = []

for line in cleaned_train_text:

    line_list = []

    for elt in line.split(' '):

        if len(elt)>0:

            line_list.append(elt)

    Sentences.append(line_list)
print(Sentences[0])