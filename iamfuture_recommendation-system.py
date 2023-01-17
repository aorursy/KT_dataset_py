# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





import gc





# Any results you write to the current directory are saved as output.
data_rating = pd.read_csv('../input/the-movies-dataset/ratings.csv')

data_rating.head()

data_rating = data_rating[:1110000]

gc.collect()

print(len(data_rating))


rating = data_rating['rating']

sn.countplot(rating)

plt.show()
from surprise import SVD, Reader, Dataset,evaluate



reader = Reader()

data = Dataset.load_from_df(data_rating[['userId','movieId','rating']], reader)



gc.collect()
data.split(n_folds=3)

gc.collect()
svd = SVD()

evaluate(svd, data, measures=['RMSE', 'MAE'])

gc.collect()
trainset = data.build_full_trainset()

svd.fit(trainset)



gc.collect()
data_rating[data_rating['userId'] == 3]
svd.predict(1, 302, 3)

#gc.collect()
os.listdir('../input')

os.listdir('../input/tmdb-movie-metadata')
tmdb_data_cred = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

print(tmdb_data_cred.columns)



print()



tmdb_data_mov = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

print(tmdb_data_mov.columns)
tmdb_data_cred.columns = ['id','tittle','cast','crew']

tmdb_data_mov = tmdb_data_mov.merge(tmdb_data_cred,on='id')
tmdb_data_mov.head()
#on a scale of 10

C = tmdb_data_mov['vote_average'].mean()

C
val = tmdb_data_mov['vote_count'].quantile(0.85)

val
#movies that qualify our 0.85 value

qual_movies = tmdb_data_mov.copy().loc[tmdb_data_mov['vote_count'] >= val]

qual_movies.shape
""" weighted rating (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C where:

R = average for the movie (mean) = (Rating)

v = number of votes for the movie = (votes)

m = minimum votes required to be listed in the Top 250 (currently 25000)

C = the mean vote across the whole report (currently 7.0)

"""



def weig_range(x,m=val,C=C):

    v = x['vote_count']

    R = x['vote_average']

    return (v / (v+m) * R) + (m / (m+v) * C)

    

qual_movies['scores'] = qual_movies.apply(weig_range,axis=1)

qual_movies = qual_movies.sort_values('scores', ascending=False)



#Print the top 10 movies

qual_movies[['title', 'vote_count', 'vote_average', 'scores']].head(10)
# TOP 10 MOVIES according to popularity



popular = tmdb_data_mov.sort_values('popularity',ascending= False)

sn.barplot(popular['popularity'].head(10),popular['title'].head(10))

plt.show()


