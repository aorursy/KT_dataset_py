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
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud, ImageColorGenerator
dataset = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')

dataset.head(10)
dataset.describe()
dataset.info()
text = ",".join(review for review in dataset.Title)

wordcloud = WordCloud(max_words=200,collocations=False,background_color="black").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
total = dataset.isnull().sum().sort_values(ascending=False)

percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
dataset.drop(['Rotten Tomatoes','Unnamed: 0','Type'],axis=1, inplace=True)
netflix_movies_count = len(dataset[dataset['Netflix'] == 1].index)

hulu_movies_count = len(dataset[dataset['Hulu'] == 1].index)

prime_movies_count =len(dataset[dataset['Prime Video'] == 1].index)

disney_movies_count = len(dataset[dataset['Disney+'] == 1].index)





print(netflix_movies_count)

print(hulu_movies_count)

print(prime_movies_count)

print(disney_movies_count)

label=['Netflix','Hulu', 'Prime Video','Disney+']

count=[netflix_movies_count,hulu_movies_count,prime_movies_count,disney_movies_count]

platform = pd.DataFrame(

    {'Platform': label,

     'MovieCount': count,

    })

platform
fig = px.pie(platform,names='Platform', values='MovieCount')

fig.update_traces(rotation=45, pull=[0.1,0.03,0.03,0.03,0.03],textinfo="percent+label", title='Movie Count per platform')

fig.show()
yearly_movie_count = dataset.groupby('Year')['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})

fig = px.bar(yearly_movie_count, x='Year', y='MovieCount', color='MovieCount', height=600)

fig.show()
dataset['Genres'].value_counts().head(5)
top_5_genres = ['Drama','Documentary','Comedy', 'Comedy,Drama','Horror']

table = dataset.loc[:,['Year','Genres','IMDb']]

table['AvgRating'] = table.groupby([table.Genres,table.Year])['IMDb'].transform('mean')

table.drop('IMDb', axis=1, inplace=True)

table = table[(table.Year>2009) & (table.Year<2020)]

table = table.loc[table['Genres'].isin(top_5_genres)]

table = table.sort_values('Year')
fig=px.bar(table,x='Genres', y='AvgRating', animation_frame='Year', 

           animation_group='Genres', color='Genres', hover_name='Genres', range_y=[0,10])

fig.update_layout(showlegend=False)

fig.show()
top_movies=dataset[dataset['IMDb']==9.3][['Title','Directors']]

top_movies
text = ",".join(review for review in top_movies.Title)

wordcloud = WordCloud(max_words=200,collocations=False,background_color="black").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
movies_by_country = dataset.groupby('Country')['Title'].count().reset_index().sort_values('Title',ascending = False).head(10).rename(columns = {'Title':'MovieCount'})

fig = px.pie(movies_by_country,names='Country', values='MovieCount')

fig.update_traces(rotation=180, pull=[0.1,0.03,0.03,0.03,0.03],textinfo="percent+label", title='Movie Count per Country')

fig.update_layout(showlegend=False)

fig.show()
longest_movies = dataset.sort_values('Runtime',ascending = False).head(10)

fig = px.bar(longest_movies, x='Title', y='Runtime', color='Runtime', height=500, title='Runtime of the top 10 longest movies')

fig.show()
runtime_top_movies=dataset.loc[dataset['IMDb']==9.3][['Title','Runtime']]

runtime_top_movies
fig = px.bar(runtime_top_movies, x='Title', y='Runtime', color='Runtime', height=700, title='Runtime of the highest rated movies')

fig.show()