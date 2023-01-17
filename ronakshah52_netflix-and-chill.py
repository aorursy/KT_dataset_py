# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

nlp = spacy.load("en_core_web_sm")

import gensim

import matplotlib.pyplot as plt

import plotly

import datetime

import plotly.graph_objects as go

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob 

import re

from collections import Counter

from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

# IMDb_titles = pd.read_csv("/kaggle/input/imdb-dataset/title.basics.tsv/title.basics.tsv", sep = "\t")

# IMDb_name = pd.read_csv("/kaggle/input/imdb-dataset/name.basics.tsv/name.basics.tsv", sep = "\t")
data.head(5)
genres = []

for row in data.listed_in.str.split(","):

    for el in row:

        genres.append(el)

genres = pd.Series(genres)

genres.shape
# bold('**MOST POPULAR GENRES ON NETFILX ARE:**')

# bold('**DOCUMENTARIES,COMEDIES, DRAMAS, INTERNATIONAL, ACTION**')

import squarify

# data['Genres'] = data['listed_in'].str.extract('([A-Z]\w{2,})', expand=True)

genres = []

for row in data.listed_in.str.replace("Movies","").str.replace("TV Shows", "").str.replace("TV", "").str.split(","):

    for el in row:

        genres.append(el.strip())

genres = pd.DataFrame(genres)

genres.columns = ['Genres']

temp_df = genres['Genres'].value_counts().reset_index()

temp_df = temp_df[temp_df['Genres'] >= 100]

sizes=np.array(temp_df['Genres'])

labels=temp_df['index']

colors = [plt.cm.Paired(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12,8), dpi= 100)

squarify.plot(sizes=sizes, label=labels, color = colors, alpha=.5, edgecolor="black", linewidth=1, text_kwargs={'fontsize':11})

plt.title('Treemap of Genres of Netflix Show', fontsize = 15)

plt.axis('off')

plt.show()

# px.treemap(temp_df, values = "Genres")
temp_df = data.query("type == 'Movie'")

temp_df['duration'] = temp_df['duration'].str.replace(" min","").str.strip()

temp_df['rating'] = temp_df['rating'].fillna("NA")

temp_df['release_year'] = temp_df['release_year'].astype('str')

temp_df.duration = temp_df.duration.astype('int')

temp_df['Genre'] = temp_df.listed_in.apply(lambda x: x.split(",")[0].replace("Movies","").replace("TV Shows","").replace("TV","").strip())

temp_df = temp_df.groupby(['Genre']).mean().reset_index()

fig = px.line_polar(temp_df,r = "duration", theta = "Genre", line_close = True, template = "plotly_dark")

fig.update_layout(title = "How Duration varies with Genre")

fig.show()
temp_df = data.query("type == 'TV Show'")

temp_df['duration'] = temp_df['duration'].str.replace(" Seasons","").str.replace(" Season","").str.strip()

temp_df.head()

# temp_df['rating'] = temp_df['rating'].fillna("NA")

# temp_df['release_year'] = temp_df['release_year'].astype('str')

# temp_df['']

temp_df.duration = temp_df.duration.astype('int')

temp_df['Genre'] = temp_df.listed_in.apply(lambda x: x.split(",")[0].replace("Movies","").replace("TV Shows","").replace("TV","").strip())

temp_df = temp_df.groupby(['Genre']).mean().reset_index()

fig = px.line_polar(temp_df,r = "duration", theta = "Genre", line_close = True, template = "plotly_dark")

fig.update_layout(title = "How Seasons varies with Genre")

# fig.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate((' '.join(data['description']).lower()))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in description',fontsize = 30)

plt.show()