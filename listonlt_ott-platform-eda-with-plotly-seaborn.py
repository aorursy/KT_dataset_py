import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

%config InlineBackend.figure_format = 'retina' 

plt.rcParams['figure.figsize'] = 8, 5

pd.options.mode.chained_assignment = None 

pd.set_option('display.max_columns',None)

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')

df.head()
print('Number of rows and columns :',df.shape) # Number of rows and columns
df.describe()
percentage_missing_values = round(df.isnull().sum()*100/len(df),2).reset_index()

percentage_missing_values.columns = ['column_name','percentage_missing_values']

percentage_missing_values = percentage_missing_values.sort_values('percentage_missing_values',ascending = False)

percentage_missing_values
sns.distplot(df['Runtime']);
sns.distplot(df['IMDb']);
movie_count_by_language = df.groupby('Language')['Title'].count().reset_index().sort_values('Title',ascending = False).head(10).rename(columns = {'Title':'Movie Count'})

fig = px.bar(movie_count_by_language, x='Language', y='Movie Count', color='Movie Count', height=600)

fig.show()
yearly_movie_count = df.groupby('Year')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'})

fig = px.bar(yearly_movie_count, x='Year', y='Movie Count', color='Movie Count', height=600)

fig.show()
movies_by_country = df.groupby('Country')['Title'].count().reset_index().sort_values('Title',ascending = False).head(10).rename(columns = {'Title':'Movie Count'})

fig = px.bar(movies_by_country, x='Country', y='Movie Count', color='Movie Count', height=600)

fig.show()
lengthiest_movies = df.sort_values('Runtime',ascending = False).head(10)

fig = px.bar(lengthiest_movies, x='Title', y='Runtime', color='Runtime', height=600)

fig.show()
digital_platforms = df[['Netflix','Hulu','Prime Video','Disney+']].sum().reset_index()

digital_platforms.columns = ['Platform', 'Movie Count']

digital_platforms = digital_platforms.sort_values('Movie Count',ascending = False)

labels = digital_platforms.Platform

values = digital_platforms['Movie Count']

pie = go.Pie(labels=labels, values=values, marker=dict(line=dict(color='#000000', width=1)))

layout = go.Layout(title='Digital Platforms Movie Share')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
top_rated_movies = df.sort_values('IMDb',ascending = False).head(10)

fig = px.bar(top_rated_movies, x='Title', y='IMDb', color='IMDb', height=600)

fig.show()
top_directors = df.groupby('Directors')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(10)

fig = px.bar(top_directors, x='Directors', y='Movie Count', color='Movie Count', height=600)

fig.show()
top_genres = df.groupby('Genres')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(10)

fig = px.bar(top_genres, x='Genres', y='Movie Count', color='Movie Count', height=600)

fig.show()