# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tmdb = pd.read_csv('/kaggle/input/tmdb-5000-movies/tmdb_5000_movies.csv')

tmdb.head()
tmdb.describe()
ax = sns.distplot(tmdb.vote_average)

ax.set(xlabel = 'Vote Average', ylabel = 'Density')

ax.set_title('Average votes - TMDB')
ax = sns.distplot(tmdb.vote_average, norm_hist = False, kde= False)

ax.set(xlabel = 'Vote Average', ylabel = 'Frequency')

ax.set_title('Average votes - TMDB')
ax = sns.boxplot(tmdb.vote_average)

ax.set(xlabel='Average Rating')

ax.set_title('Average Rating Distribution of TMDB')
tmdb.query('vote_average == 0')
tmdb_1 = tmdb.query('vote_count >= 10')

tmdb_1.describe()
ax = sns.distplot(tmdb_1.vote_average)

ax.set(xlabel = 'Vote Average', ylabel = 'Density')

ax.set_title('Average votes - TMDB')
ax = sns.distplot(tmdb_1.vote_average, norm_hist = False, kde= False)

ax.set(xlabel = 'Vote Average', ylabel = 'Frequency')

ax.set_title('Average votes - TMDB +10 Votes')
mlens = pd.read_csv("/kaggle/input/movielens-100k-small-dataset/ratings.csv")

mlens.head()
mleans_avg = mlens.groupby('movieId').mean()['rating']

mleans_avg.head()
ax = sns.distplot(mleans_avg.values)

ax.set(xlabel='Average votes', ylabel='Density')

ax.set_title('Average votes - MovieLens')
mleans_count = mlens.groupby("movieId").count()

mleans_count_10 = mleans_count.query("rating >= 10").index

mleans_avg_10 = mleans_avg.loc[mleans_count_10.values]

mleans_avg_10.head()
ax = sns.distplot(mleans_avg_10)

ax.set(xlabel='Average votes', ylabel='Density')

ax.set_title('Average votes - MovieLens >=10 ')
ax = sns.boxplot(mleans_avg_10.values)

ax.set(xlabel='Average Rating')

ax.set_title('Average Rating Distribution of MovieLens')
ax = sns.distplot(mleans_avg_10, 

                  hist_kws = {'cumulative':True}, 

                  kde_kws = {'cumulative':True})

ax.set(xlabel='Average Votes', ylabel='Cumulative Ratio')

ax.set_title('Average Rating of MovieLens')
ax = sns.distplot(tmdb_1.vote_average, 

                  hist_kws = {'cumulative':True}, 

                  kde_kws = {'cumulative':True})

ax.set(xlabel='Average Votes', ylabel='Cumulative Ratio')

ax.set_title('Average Rating of TMDB')
ax = sns.distplot(tmdb_1.vote_count)

ax.set(xlabel = 'Votes', ylabel = 'Density')

ax.set_title ('Count of votes - TMDB')
ax = sns.distplot(tmdb_1.query('budget > 0').budget)

ax.set(xlabel = 'Budget', ylabel = 'Density')

ax.set_title ('Budget on movies - TMDB')
ax = sns.distplot(tmdb_1.popularity)

ax.set(xlabel = 'Popularity', ylabel = 'Density')

ax.set_title ('Popularity on movies - TMDB')
ax = sns.distplot(tmdb_1.query('runtime > 0').runtime.fillna(tmdb_1.runtime.mean()))

ax.set(xlabel = 'Runtime', ylabel = 'Density')

ax.set_title ('Runtime on movies - TMDB')
mleans_avg_10.describe()
mleans_avg_10.head(5).mean()
avg = list()

for i in range(1, len(mleans_avg_10)):

    avg.append(mleans_avg_10[0:i].mean())

plt.plot(avg)