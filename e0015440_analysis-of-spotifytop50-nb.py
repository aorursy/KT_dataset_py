# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

plt.rcParams["patch.force_edgecolor"] = True
top50_df = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv", encoding='ISO-8859-1', index_col=0)

top50_df.head()
# info on null fields in data

top50_df.info()
cat_cols = ['Track.Name', 'Artist.Name', 'Genre']

int_cols = [name for name in top50_df.columns if top50_df[name].dtype in ['int64']]
# standardise all int64 columns to same scale

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

transformed = pd.DataFrame(scaler.fit_transform(top50_df[int_cols]), columns=int_cols, index=top50_df.index)
# join back with categorical columns

top50_scaled = top50_df[cat_cols].join(transformed)
top50_scaled.head()
top50_scaled[cat_cols].describe()
top50_scaled.describe()
# sort by popularity

top50_sorted = top50_scaled.sort_values('Popularity', ascending=False)



# Top 10 songs 

top50_sorted.head(10)
# counts of each genres in Top 50 with more than 1 song

top50_scaled['Genre'].value_counts()[top50_scaled['Genre'].value_counts()>1]
# counts of each artist in Top 50 with more than 1 song

top50_scaled['Artist.Name'].value_counts()[top50_scaled['Artist.Name'].value_counts()>1]
# Distributions and relationship between features (pairwise)

g2 = sns.PairGrid(top50_scaled[int_cols])

g2.map_offdiag(sns.regplot, ci=None)

g2.map_diag(sns.distplot, bins=10)



for axes in g2.axes.flat:

    axes.xaxis.label.set_size(15)

    axes.yaxis.label.set_size(15)
# correlations heatmap

correlations2 = top50_scaled[int_cols].corr()

plt.figure(figsize=(14,7))

sns.heatmap(data=correlations2, annot=True)
# top 10 songs

top10_songs = top50_sorted.iloc[:10,]
# get top 10 artist songs

top10_art = top10_songs['Artist.Name'].unique().tolist()



# get top 10 genre songs

top10_gen = top10_songs['Genre'].unique().tolist()
print("Artists that appeared in Top 10 songs:")

for idx, art in enumerate(top10_art):

    print("{}. {}".format(idx+1, art))

    

print()



print("Genres that appeared in Top 10 songs:")

for idx, gen in enumerate(top10_gen):

    print("{}. {}".format(idx+1, gen))
# Distributions and relationship between features (pairwise)

g = sns.PairGrid(top10_songs[int_cols])

g.map_offdiag(sns.regplot, ci=None)

g.map_diag(sns.distplot, bins=10)



for axes in g.axes.flat:

    axes.xaxis.label.set_size(15)

    axes.yaxis.label.set_size(15)
# correlations heatmap

correlations = top10_songs[int_cols].corr()

plt.figure(figsize=(14,7))

sns.heatmap(data=correlations, annot=True)
from textblob import TextBlob

from wordcloud import WordCloud, STOPWORDS

import nltk

from nltk.corpus import stopwords

import string
# stopwords 

stop_words_en = set(stopwords.words("english"))

stop_words_es = set(stopwords.words("spanish"))



# punctuations

punctuations = list(string.punctuation)
## tokenising

titles = top50_scaled['Track.Name'].map(TextBlob)



# print first 5 tokenised titles

for i in range(5):

    print(titles.iloc[i].words)
sentiments = {}

for i in range(titles.shape[0]):

    sentiments[top50_scaled['Track.Name'].iloc[i]] = titles.iloc[i].sentiment.polarity

    

sentiments = pd.DataFrame(sentiments.values(), index=top50_scaled['Track.Name'], columns=['sentiment'])

sentiments.reset_index(drop=False, inplace=True)

sentiments.head()
# descriptive stats on sentiments

sentiments.describe()
sns.distplot(sentiments['sentiment'], kde=False, bins=6)

plt.title("Distribution of sentiment of top 50 song titles")
title_str = top50_scaled['Track.Name'].map(nltk.word_tokenize)



# text cleaning - lower caps, stopwords, punctuations

for i in range(title_str.shape[0]):

    title_str.iloc[i] = [w.lower() for w in title_str.iloc[i]]

    title_str.iloc[i] = [w for w in title_str.iloc[i] if w not in punctuations]

    title_str.iloc[i] = [w for w in title_str.iloc[i] if w not in stop_words_en]

    title_str.iloc[i] = [w for w in title_str.iloc[i] if w not in stop_words_es]



# forms long paragraph of string for wordcloud

long_titles = ""

for i in range(title_str.shape[0]):

    temp = " ".join(title_str.iloc[i])

    long_titles = long_titles + " " + temp

    

# remove leading and trailing whitespaces

long_titles = long_titles.strip()

long_titles = long_titles.replace('feat', '').replace('ft.', '')

print(long_titles)
plt.figure(figsize=(12,8))

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1000,

                      height=1000).generate(long_titles)

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()