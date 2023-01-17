import pandas as pd

import numpy as np

import re

import nltk

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import bokeh.io

#from bokeh.charts import Donut, HeatMap, Histogram, Line, Scatter, show, output_notebook, output_file

from bokeh.plotting import figure

import string

import gensim.models.word2vec as w2v

import multiprocessing

import os

import re

import sklearn

import pprint

import seaborn as sns

import wordcloud

%matplotlib inline

stop = stopwords.words('english')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/billboard_lyrics_1964-2015.csv",encoding='latin-1').dropna()
#Count data

len(data)
#Show first line of dataset

data.head(1)
data['Cleaned_Lyrics'] = data['Lyrics'].str.lower().str.split()

data['Cleaned_Lyrics'] = data['Cleaned_Lyrics'].apply(lambda x : [item for item in x if item not in stop])



from nltk.tag import pos_tag

def Getting_NN(sentence):

   sentence=sentence.lower()

   cleaned=' '.join([w for w in sentence.split() if not w in stop]) 

   cleaned=' '.join([w for w , pos in pos_tag(cleaned.split()) if (pos == 'NN'  )])

   cleaned=cleaned.strip()

   return cleaned

data['pos_tag_NN']= data['Lyrics'].apply(lambda x:Getting_NN(x))



def Getting_JJR(sentence):

   sentence=sentence.lower()

   cleaned=' '.join([w for w in sentence.split() if not w in stop]) 

   cleaned=' '.join([w for w , pos in pos_tag(cleaned.split()) if (pos == 'JJR'  )])

   cleaned=cleaned.strip()

   return cleaned



data['pos_tag_JJR']= data['Lyrics'].apply(lambda x:Getting_JJR(x))
data['ly_count'] = data['Lyrics'].str.split(" ").str.len()

data['NN_count'] = data['pos_tag_NN'].str.split(" ").str.len()

data['JJR_count'] = data['pos_tag_JJR'].str.split(" ").str.len()

data.head(1)
Artist_Count = data.Artist.value_counts()[:25]

Artist_Count
plt.figure(figsize=(13,9))

plt.title("Maximum Lyrics Count By Artist",fontsize=20)

data['Artist'].value_counts()[:30].plot('bar',color='purple')
Total_year_count = data.groupby(['Year'])['ly_count'].sum()

plt.figure(figsize=(12,9))

plt.title("Lyrics Vocabulary Count Vs Year",fontsize=20)

Total_year_count.plot(kind='line',color="Red")
plt.figure(figsize=(13,9))

plt.title("Vocabulary Count In Lyrics vs year",fontsize=20)

Total_year_count.plot(kind='bar',label="Lyrics count vs year",color = 'slateblue')
Noun_year_count = data.groupby(['Year'])['NN_count'].sum()

plt.figure(figsize=(13,9))

plt.title("Artist Noun Usage In Lyrics",fontsize=20)

Noun_year_count.plot(label='Noun count vs year',kind='line',color = 'c')
Adj_year_count = data.groupby(['Year'])['JJR_count'].sum()

plt.figure(figsize=(13,9))

plt.title("Artist Adjective Usage Count In Lyrics",fontsize=20)

Adj_year_count.plot(label='Adjective count vs year',kind='line',color = 'r')
plt.figure(figsize=(12,5))

plt.title("Max Repeated Words In Lyrics",fontsize=20)

words = pd.Series(' '.join(data['Cleaned_Lyrics'].astype(str)).lower().split(" ")).value_counts()[:500]

words.plot(color='darkcyan')
allwords = ' '.join(data['Song']).lower().replace('c', '')

cloud = wordcloud.WordCloud(background_color='black',

                            max_font_size=100,

                            width=1000,

                            height=500,

                            max_words=100,

                            relative_scaling=.5).generate(allwords)

plt.figure(figsize=(15,5))

plt.axis('off')



plt.imshow(cloud);
from sklearn.feature_extraction.text import TfidfVectorizer

text = data['Lyrics']

#define vectorizer parameters

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=20000,

                                 min_df=3, stop_words='english',

                                 use_idf=True,ngram_range=(2,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(text) 

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
#Bigrams



from sklearn.feature_extraction.text import CountVectorizer

word_vectorizer = CountVectorizer(ngram_range=(2,2),max_features=25,stop_words='english',analyzer ='word',strip_accents='ascii')

bigrams=word_vectorizer.fit(data['Lyrics']).vocabulary_

bigrams
#Tigrams



tri = CountVectorizer(ngram_range=(3,3),max_features=25,stop_words='english',analyzer ='word',strip_accents='ascii')

trigrams=tri.fit(data['Lyrics']).vocabulary_

trigrams