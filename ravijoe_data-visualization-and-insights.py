# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from IPython.display import Image, HTML

import json

import datetime

import ast

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from wordcloud import WordCloud, STOPWORDS

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')



sns.set_style('whitegrid')

sns.set(font_scale=1.25)

pd.set_option('display.max_colwidth', 50)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
df1.columns

df1.columns = ['id','tittle','cast','crew']

df2= df2.merge(df1,on='id')
df2.shape

df2.columns

df2.head()

df2.info()

df2[df2['original_title'] != df2['title']][['title', 'original_title']].head()

df2 = df2.drop('original_title', axis=1)

df2[df2['revenue'] == 0].shape

df2['budget'] = pd.to_numeric(df2['budget'], errors='coerce')

df2['budget'] = df2['budget'].replace(0, np.nan)

df2[df2['budget'].isnull()].shape
df2['return'] = df2['revenue'] / df2['budget']

df2[df2['return'].isnull()].shape
df2['production_countries']

df2['year'] = pd.to_datetime(df2['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
df2['title'] = df2['title'].astype('str')

df2['overview'] = df2['overview'].astype('str')
title_corpus = ' '.join(df2['title'])

overview_corpus = ' '.join(df2['overview'])
# Create and generate a word cloud image:

wordcloud = WordCloud().generate(title_corpus)

plt.figure(figsize=(15,15))# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud().generate(overview_corpus)

plt.figure(figsize=(15,15))# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
plt.figure(figsize=(10, 10))

sns.heatmap(df2.corr())

plt.show()
from scipy.stats import pearsonr



sns.jointplot(x='budget', y='revenue', stat_func =pearsonr,data=df2)
sns.jointplot(x='popularity', y='revenue', stat_func =pearsonr,data=df2)

sns.jointplot(x='vote_count', y='revenue', stat_func =pearsonr,data=df2)

df2[['title', 'vote_count', 'year']].sort_values('vote_count', ascending=False).head(10)

most_popular_movies_and_return = df2[['title', 'vote_count', 'year','revenue','return']].sort_values('vote_count', ascending=False)
most_popular_movies_and_return=most_popular_movies_and_return[most_popular_movies_and_return['revenue']>=1000000000]
# sns.catplot(x="Surface", data=injury,hue='BodyPart', kind="count")

chart = sns.catplot(x="year", data=most_popular_movies_and_return, kind="count")

chart.set_xticklabels(rotation=45)



plt.gcf().set_size_inches(24, 12)
df2[df2['vote_count'] > 3000][['title', 'vote_average', 'vote_count' ,'year']].sort_values('vote_average', ascending=False).head(10)
sns.jointplot(x='vote_average', y='popularity', stat_func =pearsonr,data=df2)

df2['genres'] = df2['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
s = df2.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'
gen_df = df2.drop('genres', axis=1).join(s)

df2.head()
pop_gen = pd.DataFrame(gen_df['genre'].value_counts()).reset_index()

pop_gen.columns = ['genre', 'movies']

pop_gen.head(10)
plt.figure(figsize=(18,8))

sns.barplot(x='genre', y='movies', data=pop_gen.head(15))

plt.show()
res = df2.set_index(['popularity', 'revenue'])['genres'].apply(pd.Series).stack()

res = res.reset_index()

res.columns = ['popularity','revenue','sample_num','genre']

res.drop('sample_num',axis=1,inplace=True)

res
res_400=res[res['revenue']>=400000000]

df2[df2['budget'].notnull()][['title', 'budget', 'revenue', 'return', 'year']].sort_values('budget', ascending=False).head(10)
gross_top = df2[['title', 'budget', 'revenue', 'year']].sort_values('revenue', ascending=False).head(10)

gross_top


plt.figure(figsize=(18,5))

year_revenue = df2[(df2['revenue'].notnull()) & (df2['year'] != 'NaT')].groupby('year')['revenue'].max()

# plt.plot(year_revenue.index, year_revenue)

# chart = plt.plot(year_revenue.index, year_revenue)

# chart.set_xticklabels(rotation=45)

# plt.gcf().set_size_inches(16, 8)

# plt.show()

year_revenue.plot(kind='line')
s = df2.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'
gen_df = df2.drop('genres', axis=1).join(s)

pop_genre = pd.DataFrame(gen_df['genre'].value_counts()).reset_index()

pop_genre.columns = ['genre', 'movies']

pop_gen.head(10)
plt.figure(figsize=(18,8))

sns.barplot(x='genre', y='movies', data=pop_gen.head(15))

plt.show()