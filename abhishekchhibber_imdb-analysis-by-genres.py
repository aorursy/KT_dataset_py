import pandas as pd

import numpy as np

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv('../input/movie_metadata.csv')

data.head()
data['profit'] = data['gross'] - data['budget']
data.columns
dt = data[['genres','movie_title','language','title_year','imdb_score','duration','profit']]

dt.head()
dt.shape
s = dt['genres'].str.split('|').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'genres'

del dt['genres']

df = dt.join(s)
df.head()
df.shape
df['genres'].unique()
len(df['genres'].unique())
df1 = df[df['imdb_score']>=7]
df1.head()
df2 = (pd.DataFrame(df1.groupby('genres').movie_title.nunique())).sort_values('movie_title', ascending=False )
df2
df2[['movie_title']].plot.barh(stacked=True, title = 'Genres with >= 7 ratings', figsize=(8, 8));
df3 = df[['movie_title', 'profit','genres']]
df3.head()
# Checking for NaN

df3.loc[df3['genres'] == 'News']
df4 = df3.groupby(['genres']).mean()
df4['profit_million'] = df4['profit']/1000000

del df4['profit']
df4.sort_values('profit_million', ascending=False, inplace = True )
df4[['profit_million']].plot.barh(stacked=True, title = 'Genres by profit (US$ million)', figsize=(8, 8));
df5 = df[['movie_title', 'duration','genres']]
df5.head()
df6 = df5.groupby(['genres']).mean()

df6['average_duration']  = df6['duration'].round(2)

del df6['duration']

df6.sort_values('average_duration', ascending=False, inplace = True )
df6
df6[['average_duration']].plot.barh(stacked=True, title = 'Average Duration by Genre (minutes)', figsize=(8, 8));
df7 = df[['title_year','genres']]
df7.head()
df7.shape
df8 = df7[df7['title_year']>2005]
df8.shape
df9 = df8[df8['genres'] == 'Thriller']
df9.shape
df10 = df9.groupby(['title_year']).count()
df10
df10[['genres']].plot.barh(stacked=True, title = 'Thrillers Released (By Year)', figsize=(8, 8));
df11 = df8[df8['genres'] == 'Mystery']
df11.shape
df12 = df11.groupby(['title_year']).count()
df12
df10[['genres']].plot.barh(stacked=True, title = 'Mysteries Released (By Year)', figsize=(8, 8));