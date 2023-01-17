# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import json

import ast

from wordcloud import WordCloud, STOPWORDS

import plotly.express as px 

import plotly.offline as py

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


movies=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

ratings=pd.read_csv('/kaggle/input/the-movies-dataset/ratings.csv')
movies.head().transpose()
movies.describe()
movies.info()
missing_features=pd.DataFrame(movies.isnull().mean().sort_values(ascending=False),columns=['missing_Ratio'])

missing_features[missing_features['missing_Ratio']>0]
movies['Year']=pd.to_datetime(movies['release_date'],errors='coerce').apply(lambda x:x.year)

movies['month']=pd.to_datetime(movies['release_date'],errors='coerce').apply(lambda x:x.month )
movies['budget']=pd.to_numeric(movies['budget'],errors='coerce')
movies['return']=movies['revenue'].replace(0,np.nan)/movies['budget'].replace(0,np.nan)
movies['genres']=movies['genres'].apply(ast.literal_eval)

#movies['genres']=movies['genres'].apply(lambda x:[d['name'] for d in x] if isinstance(x,list) else[])
movies['genres']=movies['genres'].apply(lambda x:[d['name'] for d in x] if isinstance(x,list) else[])
df_genres=movies['genres'].apply(pd.Series).stack().reset_index(level=1,drop=True)
df_genres.name='gen'

df_genres=movies.join(df_genres)
df_genres.gen.value_counts(ascending=False).plot(kind='bar',figsize=(15,5))
df_genres['count']=1
top_5_genres=df_genres['gen'].value_counts(ascending=False).index[:5]

pd.pivot_table(index='Year',columns='gen',values='count',aggfunc=sum,data=df_genres[df_genres['gen'].apply(lambda x: x in top_5_genres)]).plot(figsize=(15,5))

plt.legend(bbox_to_anchor=[1.1,1])

for i in range(2010,2021):

    print(i,':',movies[movies['Year']==i].shape[0])
movies['production_countries']=movies['production_countries'].fillna('[]').apply(ast.literal_eval)

movies['production_countries']=movies['production_countries'].apply(lambda x:[d['name'] for d in x] if isinstance(x,list) else[])
countries=movies['production_countries'].apply(pd.Series).stack().reset_index(level=1,drop=True)
countries.name='countries'

df_countries=movies.join(countries)
countries.value_counts(ascending=False)[:20].plot(kind='bar',figsize=(15,5))
c=countries.value_counts(ascending=False)[1::]

fig = px.choropleth( locations=c.index,

                    locationmode='country names',

                    color=c.values,

                    hover_name=c.index, 

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
df_countries['count']=1

top_10_countries=c[:10].index

pd.pivot_table(index='Year',columns='countries',values='count',aggfunc=sum,data=df_countries[df_countries['countries'].apply(lambda x : x in top_10_countries)]).plot(figsize=(15,5))

plt.legend(bbox_to_anchor=[1.1,1])

plt.title('Production per country througout the Year (US not included)')
df_countries[df_countries['countries']=='United States of America'].groupby('Year')['count'].sum().plot(figsize=(14,5))
movies.head().transpose()
movies['production_companies'] = movies['production_companies'].fillna('[]').apply(ast.literal_eval)

df_companies=movies['production_companies'].apply(lambda x:[d['name'] for d in x] if  isinstance(x,list) else [])

df_companies=df_companies.apply(pd.Series).stack().reset_index(level=1,drop=True)

df_companies.name='companies'
df_companies=movies.drop('production_companies',axis=1).join(df_companies)
df_companies['companies'].value_counts(ascending=False)[:20].plot(kind='bar',figsize=(15,5))
df_companies['count']=1

top_5_companies=df_companies['companies'].value_counts(ascending=False)[:5].index

pd.pivot_table(index='Year',columns='companies',values='count',aggfunc=sum,data=df_companies[df_companies['companies'].apply(lambda x : x in top_5_companies)]).plot(figsize=(15,5))

plt.legend(bbox_to_anchor=[1.1,1])
df_companies.groupby(by='companies')['revenue'].mean().sort_values(ascending=False)[:20].plot(kind='bar',figsize=(15,5))
df_companies.groupby(by='companies')['budget'].mean().sort_values(ascending=False)[:20].plot(kind='bar',figsize=(15,5))
df_companies.groupby(by='companies')['return'].mean().sort_values(ascending=False)[:20].plot(kind='bar',figsize=(15,5))
movies['belongs_to_collection'] = movies['belongs_to_collection'].fillna('[]').apply(ast.literal_eval)

df_collection=movies['belongs_to_collection'].apply(lambda x:x['name'] if  isinstance(x,dict) else [])
df_collection=df_collection.apply(pd.Series).stack().reset_index(level=1,drop=True)
df_collection.name='collection'
df_collection=movies.drop('belongs_to_collection',axis=1).join(df_collection)
df_collection['collection'].value_counts(ascending=False)[:10].plot(kind='bar',figsize=(15,5))
df_collection.groupby(by='collection')['revenue'].sum().sort_values(ascending=False)[:10].plot(kind='bar',figsize=(15,5))
df_collection.groupby(by='collection')['budget'].sum().sort_values(ascending=False)[:10].plot(kind='bar',figsize=(15,5))
movies.info()
def clean_numeric(x):

    try:

        return float(x)

    except:

        return np.nan
movies['popularity']=movies['popularity'].apply(clean_numeric)
movies.columns
movies[['title','Year','popularity']].sort_values(by='popularity',ascending=False)
movies['vote_count']=movies['vote_count'].apply(clean_numeric)

movies['vote_average']=movies['vote_average'].apply(clean_numeric)
movies[['title','vote_average','vote_count']].sort_values('vote_count',ascending=False)
title_corpus=' '.join(movies['title'].dropna())

overview_corpus=' '.join(movies['overview'].dropna())
title_worldcloud=WordCloud(stopwords=STOPWORDS,background_color='white',height=2000,width=4000).generate(title_corpus)

plt.figure(figsize=(17,5))

plt.axis('off')

plt.imshow(title_worldcloud)

plt.show()
overview_worldcloud=WordCloud(stopwords=STOPWORDS,background_color='white',height=2000,width=4000).generate(overview_corpus)

plt.figure(figsize=(17,5))

plt.axis('off')

plt.imshow(overview_worldcloud)

plt.show()
pd.DataFrame(movies['original_language'].value_counts(ascending=False))
movies['original_language'].value_counts(ascending=False)[:10].plot(kind='bar',figsize=(15,10))

plt.title('Top 10 Most popular languages')
df_countries.groupby('original_language')['countries'].nunique().sort_values(ascending=False)[:10].plot(kind='bar',figsize=(15,5))

plt.title('Number of Producing countries per language')
df_countries[df_countries['countries']=='Germany']['original_language'].value_counts().head()
df_countries[df_countries['countries']=='Italy']['original_language'].value_counts().head()
df_countries[df_countries['countries']=='Japan']['original_language'].value_counts().head()
movies['spoken_languages']=movies['spoken_languages'].fillna('[]').apply(ast.literal_eval)

movies['spoken_languages']=movies['spoken_languages'].apply(lambda x:[d['name'] for d in x] if isinstance(x,list) else [])
df_languages=movies['spoken_languages'].apply(pd.Series).stack().reset_index(level=1,drop=True)
df_languages.value_counts()[:10]
movies.groupby('month')['adult'].count().plot(figsize=(10,5))

plt.title('Movies per month 1870-2020')
movies[movies['Year']>2000].groupby('month')['adult'].count().plot(figsize=(10,5))

plt.title('Movies per month 2000-2020')
movies['count']=1

df_year_month=pd.pivot_table(columns='Year',index='month',values='count',aggfunc=sum,data=movies[movies['Year']>2000])



#sns.set(font_scale=1)

f, ax = plt.subplots(figsize=(16, 8))

sns.heatmap(df_year_month, annot=True, linewidths=.5, fmt='n',ax=ax)
movies[movies['Year']<=2015].groupby('Year')['budget'].mean().plot(figsize=(14,5))
movies[movies['Year']<=2015].groupby('Year')['revenue'].mean().plot(figsize=(14,5))
sns.jointplot(x=movies['budget'],y=movies['revenue'])
df_genres.groupby(by='gen')['budget'].mean().sort_values(ascending=False)[:10].plot(kind='bar',figsize=(13,5))
df_genres.groupby(by='gen')['revenue'].mean().sort_values(ascending=False)[:10].plot(kind='bar',figsize=(13,5))
movies[['original_title','revenue']].sort_values('revenue',ascending=False)[:10]
movies[['original_title','budget']].sort_values('budget',ascending=False)[:10]