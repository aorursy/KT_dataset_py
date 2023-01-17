# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

# Read csv dataset using pandas, see docs here:
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html?highlight=csv#pandas.read_csv
# Use only Lyrics and Year cols (3 and 4)
def ingest_train():
    data = pd.read_csv('../input/billboard_lyrics_1964-2015.csv', encoding='latin-1', usecols=[3,4])
    return data
data = ingest_train()
# stats about the dataset
data.describe()
# data headers (column names) with some rows for context
data.head()
# Import some utils to clean the data
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
# Stopwords and punctuation
from nltk.tokenize import WordPunctTokenizer
import re
tokenizer = WordPunctTokenizer()

def clean_data(text):
        lower_case = text.lower() if (type(text) is str) else text
        tokens = tokenizer.tokenize(lower_case)
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [get_lemma(token) for token in tokens]
        return tokens


# Clean lyrics for each song
text_data = []
for index, song in enumerate(data.Lyrics):
    if (type(song) is str): 
        song = clean_data(song)
        text_data.append(song)
        


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from wordcloud import WordCloud, STOPWORDS


popular_words = []
for t in data.Lyrics:
    popular_words.append(t)
popular_words = pd.Series(popular_words).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color='white').generate(popular_words)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]


import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

# Find 5 topics
import gensim
NUM_TOPICS = 5
#ignore warnings for deprecated numpy function in lda
import warnings

topics = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')

    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)


dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(lda_display)
