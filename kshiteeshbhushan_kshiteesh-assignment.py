# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

import numpy as np

import scipy as sp

import sklearn

import sys

from nltk.corpus import stopwords, wordnet

import nltk

from nltk.stem import WordNetLemmatizer

from gensim.models import ldamodel

from gensim.models.hdpmodel import HdpModel

from gensim.models import CoherenceModel

from gensim import matutils, models

import gensim.corpora

from sklearn.decomposition import NMF;

from sklearn.preprocessing import normalize;

import scipy.sparse

import string

import pickle;

import re;

from nltk import pos_tag, word_tokenize

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
df = pd.read_csv('../input/unstructured-l0-nlp-hackathon/data.csv')
df.head()
def cleaning(text):

    text = re.sub(r'[.,"?!:/[;\)/(*%&^]',' ',text)

    stopwords_list = stopwords.words('english')

    text = ' '.join([x.lower() for x in text.split() if x not in stopwords_list and x.isalpha() and len(x)>2])

    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",text).split())

    return(text)

df['clean_text'] = df['text'].apply(cleaning)
df['tokens'] = df['clean_text'].apply(lambda x:[token for token, pos in nltk.pos_tag(nltk.word_tokenize(x)) if pos.startswith('JJ') or pos.startswith('NN')])
def lemmatizing(tokens):

    lemms = []

    for tok in tokens:

        lemms.append(lemmatizer.lemmatize(tok))

    return lemms
df['lemmatized'] = df['tokens'].apply(lemmatizing)
data = df['tokens'].to_list()

data_list = []

for sublists in data:

    for items in sublists:

        data_list.append(items)
num_topics = 5

id2word = gensim.corpora.Dictionary(data)

corpus = [id2word.doc2bow(text) for text in data]

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,random_state=1,passes=50,eta = 1)
def get_lda_topics(model, num_topics):

    word_dict = {}

    topics = model.show_topics(num_topics,20)

    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \

                 for i,words in lda.show_topics(5,20)}

    return pd.DataFrame.from_dict(word_dict)
get_lda_topics(lda, num_topics)
score = lda[corpus]
final_topic = []

for i in range(len(score)):

    v=dict(score[i])

    for topic, scores in v.items():

        if scores == max(v.values()):

            final_topic.append(topic)
df["topic"] = final_topic

df.head()
df[['Id','topic']].to_csv('results.csv',index=False)