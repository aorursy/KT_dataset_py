# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/movie_metadata.csv')
df.color.unique()
df.columns
df_gross = df.groupby('director_name')['gross'].sum()

df_gross.sort_values(ascending=False).head(20)
df_gross_actor1 = df.groupby('actor_1_name')['gross'].sum()

df_gross_actor1.sort_values(ascending=False).head(20)
df.ix[10,:]
df_compressed_rating = df[df['imdb_score'] >=2 ]

df_compressed_rating.imdb_score.min()
df_groupby_ratings = df_compressed_rating.groupby(['imdb_score'])['movie_title'].count()
import matplotlib.pyplot as plt
%matplotlib inline
df_groupby_ratings.plot()
df_movies_year = df.groupby(['title_year'])['movie_title'].count()

df_movies_year.head()
df_movies_year.plot('line')
df_movies_year_gross = df.groupby(['title_year'])['gross'].sum()

df_movies_year_gross.plot()
df_movies_year_budget = df.groupby(['title_year'])['budget'].sum()

df_movies_year_budget.plot()
turnover_per_movie_year = df_movies_year_gross / df_movies_year

turnover_per_movie_year.plot()
budget_per_movie_year = df_movies_year_budget / df_movies_year

budget_per_movie_year.plot()
turnover_per_movie_year[turnover_per_movie_year.index > 1980].plot()
df_movies_year_duration = df.groupby(['title_year'])['duration'].sum()

df_movies_year_avg_duration = df_movies_year_duration / df_movies_year

df_movies_year_avg_duration.plot()
f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(turnover_per_movie_year[turnover_per_movie_year.index > 1950])

axarr[0].set_title('Movie Length vs. Gross Revenue Trend')

axarr[1].plot(df_movies_year_avg_duration[df_movies_year_avg_duration.index > 1950])
#f, axarr = plt.subplots(2, sharex=True)

plt.plot(budget_per_movie_year[budget_per_movie_year.index > 1950])

#axarr[0].set_title('Budget vs. Gross Revenue Trend')

plt.plot(turnover_per_movie_year[turnover_per_movie_year.index > 1950], 'r')
plt.plot(turnover_per_movie_year[turnover_per_movie_year.index > 1970]

         - budget_per_movie_year[turnover_per_movie_year.index > 1970])
df_highest_rate = df.sort_values(by = 'imdb_score', ascending=False)
g = df_highest_rate.groupby('imdb_score').apply(lambda x: x.head(2))

h = g.set_index('imdb_score').reset_index()
top_rated = h[h['imdb_score'] >= 8].sort_values(by = 'imdb_score', ascending = False)

top_rated = top_rated.dropna()

top_rated.loc[:,['movie_title', 'director_name', 'gross', 'actor_1_name', 'actor_2_name', 'imdb_score']]
df_top_grossing_actors = df.groupby('actor_1_name')['gross'].mean()
df_top_grossing_actors.sort_values(ascending=False).head()
list(top_rated['actor_1_name'])
res = df_top_grossing_actors[df_top_grossing_actors.index.isin(list(top_rated['actor_1_name']))]

res.sort_values(ascending=False).plot('bar')
df_top_grossing_directors = df.groupby('director_name')['gross'].mean()

df_top_grossing_directors.sort_values(ascending=False).head()
res_directors = df_top_grossing_directors[df_top_grossing_directors.index.isin(list(top_rated['director_name']))]

res_directors.sort_values(ascending=False).plot('bar')