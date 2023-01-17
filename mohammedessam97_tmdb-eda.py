import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import json 

%matplotlib inline 
movies  = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
movies.head()
movies.info()
movies.hist(figsize=(10,8))
movies.shape


movies.drop(['homepage','tagline'],axis=1,inplace=True)
movies.dropna(inplace=True)
movies.drop_duplicates(inplace=True)
movies=movies[movies.revenue>0]
movies.shape
movies['year'] = pd.DatetimeIndex(movies['release_date']).year
movies.head()
plt.plot(movies.groupby('year').sum()['revenue'])
movies.head()
movies['genres']=[[item['name']for item in json.loads(genres)] for genres in movies['genres']]
genres_data = movies[['original_title','genres','popularity','year']]
genres_data=pd.DataFrame([

    [P, t, i,y] for P, T, i,y in genres_data.values

    for t in T

], columns=genres_data.columns)
genres_data.sort_values(by=['year'],inplace=True)
genres_data = genres_data[genres_data['year']>2005]
genres_data_grouped=genres_data.groupby(['year','genres']).sum()['popularity']
genres_data_grouped=genres_data_grouped.reset_index()
genres_data_grouped=genres_data_grouped.groupby(['year']).max().reset_index()
plt.figure(figsize=(20,10))

sns.barplot(x='year',y='popularity',hue='genres',data=genres_data_grouped)