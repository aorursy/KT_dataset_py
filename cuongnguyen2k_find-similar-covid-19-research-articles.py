import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import glob

import json

import re

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))


path = '/kaggle/input/'

all_json = glob.glob(f'{path}/**/*.json', recursive=True)

class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

covid_df = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])

covid_df.drop_duplicates(['abstract'], inplace=True)

covid_df.head()
covid_df['body_text'] = covid_df['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

covid_df['abstract'] = covid_df['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))



def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



covid_df['body_text'] = covid_df['body_text'].apply(lambda x: lower_case(x))

covid_df['abstract'] = covid_df['abstract'].apply(lambda x: lower_case(x))

covid_df.head()
text = covid_df.drop(["paper_id", "abstract"], axis=1)

text.head()

text.to_csv('./clean_text.csv')
import spacy

spacy.load('en')

from spacy.lang.en import English

parser = English()
def tokenize(text):

    lda_tokens = []

    tokens = parser(text)

    for token in tokens:

        if token.orth_.isspace():

            continue

        elif token.like_url:

            lda_tokens.append('URL')

        elif token.orth_.startswith('@'):

            lda_tokens.append('SCREEN_NAME')

        else:

            lda_tokens.append(token.lower_)

    return lda_tokens
import nltk

from nltk.corpus import wordnet as wn

from nltk.stem.wordnet import WordNetLemmatizer



def get_lemma(word):

    lemma = wn.morphy(word)

    if lemma is None:

        return word

    else:

        return lemma

def get_lemma2(word):

    return WordNetLemmatizer().lemmatize(word)
en_stop = set(nltk.corpus.stopwords.words('english'))
def prepare_text_for_lda(text):

    tokens = tokenize(text)

    tokens = [token for token in tokens if len(token) > 4]

    tokens = [token for token in tokens if token not in en_stop]

    tokens = [get_lemma(token) for token in tokens]

    return tokens
import random

from random import randint



text_data = []

with open('./clean_text.csv') as f:

    for line in f:

        tokens = prepare_text_for_lda(line)

        value = randint(0, 100)

        if value==99:

            text_data.append(tokens)
from gensim import corpora

dictionary = corpora.Dictionary(text_data)

corpus = [dictionary.doc2bow(text) for text in text_data]

import pickle

pickle.dump(corpus, open('corpus.pkl', 'wb'))

dictionary.save('dictionary.gensim')
import gensim

NUM_TOPICS = 10

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)

ldamodel.save('model10.gensim')

topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')

corpus = pickle.load(open('corpus.pkl', 'rb'))

lda = gensim.models.ldamodel.LdaModel.load('model10.gensim')

import pyLDAvis.gensim

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display)