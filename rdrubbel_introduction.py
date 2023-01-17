import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)t 

import json



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')



movies_df = pd.read_csv('../input/tmdb_5000_movies.csv')

credits_df = pd.read_csv('../input/tmdb_5000_credits.csv')

movies_df.info()

credits_df.info()

del credits_df['title']
movies_df.tail()
df = pd.concat([movies_df, credits_df], axis=1)

df.tail()

df['id'].equals(df['movie_id'])
newCols = ['id','title','release_date','vote_average','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status']



df2 = df[newCols]
df2.tail(2)
df2.plot.scatter ('budget', 'vote_average')
df2.plot.scatter ('vote_count', 'vote_average')
df2.plot.scatter ('budget', 'revenue')
df2.plot.scatter('vote_average','vote_count')

df2.plot.scatter('vote_average','budget')

df2.plot.scatter('vote_average','revenue')



df2.plot.scatter('vote_count','budget')

df2.plot.scatter('vote_count','revenue')

df2.plot.scatter('revenue','budget')

df2.plot.scatter('runtime', 'revenue')

df2.plot.scatter('runtime', 'vote_average')
sns.regplot(df2['vote_average'],df2['budget'])
dataframe.describe()
df.describe()
df2.describe()
