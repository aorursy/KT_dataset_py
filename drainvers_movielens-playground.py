import os



def list_all_files_in(dirpath):

    for dirname, _, filenames in os.walk(dirpath):

        for filename in filenames:

            print(os.path.join(dirname, filename))



list_all_files_in('/kaggle/input')
# Dataframes

import pandas as pd



# Linear algebra

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# List shifting

from collections import deque



# Similarities between vectors

from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from sklearn.feature_extraction.text import TfidfVectorizer



# Recommender library

import surprise as sp

from surprise.model_selection import cross_validate, train_test_split



# Sparse matrices

from scipy.sparse import coo_matrix



# LightFM

from lightfm import LightFM

from lightfm.evaluation import precision_at_k



# Stacking sparse matrices

from scipy.sparse import vstack



# Displaying stuff

from IPython.display import display



import warnings; warnings.simplefilter('ignore')
ratings = pd.read_csv('/kaggle/input/movielens100k/ratings.csv')

ratings.head(10)
reader = sp.Reader()

data = sp.Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
movies_df = pd.read_csv('../input/movielens100k/movies.csv')



# Split the year in movie titles into a separate column

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)

movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)



movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())



# Convert genres into a list

movies_df['genres'] = movies_df.genres.str.split('|')



movies_df.head()
movies_df.isna().sum()
movies_df[movies_df.isnull().any(axis=1)]
movies_df['year'].fillna(0, inplace=True)

movies_df.loc[[8505, 8507, 9017, 9063, 9118, 9124]]
movies_with_genres = movies_df.copy(deep=True)



x = []

for index, row in movies_df.iterrows():

    x.append(index)

    for genre in row['genres']:

        movies_with_genres.at[index, genre] = 1



movies_with_genres.fillna(0.0, inplace=True)

movies_with_genres.drop('genres', axis=1, inplace=True)
svd = sp.SVD()

# trainset, testset = train_test_split(data, test_size=.2)

trainset = data.build_full_trainset()

svd.fit(trainset)
# results = svd.test(testset)

# sp.accuracy.rmse(results)

# sp.accuracy.mae(results)

# sp.accuracy.fcp(results)
def collaborative_recommender(uid, n):

    reclist = movies_df.copy(deep=True)

    reclist['est'] = reclist['movieId'].apply(lambda x: svd.predict(uid, x).est)

    reclist = reclist.sort_values('est', ascending=False)

    reclist.set_index('movieId', drop=True, inplace=True)

    return reclist.head(n)
collaborative_recommender(7, 10)
def get_user_movie_ratings(uid):

    user_ratings = ratings[ratings['userId'] == uid]

    user_movie_ratings = pd.merge(movies_df, user_ratings, on='movieId')[['movieId', 'title', 'rating']]

    return user_movie_ratings
def get_user_genres(uid):

    user_movie_ratings = get_user_movie_ratings(uid)

    user_genres = movies_with_genres[movies_with_genres['movieId'].isin(user_movie_ratings['movieId'])]

    user_genres.reset_index(drop=True, inplace=True)

    user_genres.drop(['movieId', 'title', 'year'], axis=1, inplace=True)

    return user_genres
def content_based_recommender(uid, n):

    # Build user profile

    user_movie_ratings = get_user_movie_ratings(uid)

    user_genres_df = get_user_genres(uid)

    

    # Get content-based recommendations (weighted average of genres)

    user_profile = user_genres_df.T.dot(user_movie_ratings['rating'])

    # display(user_profile / user_genres_df.sum())

    genres_df = movies_with_genres.copy(deep=True).set_index(movies_with_genres['movieId']).drop(['movieId', 'title', 'year'], axis=1)

    recommendation_df = (genres_df.dot(user_profile)) / user_profile.sum()

    recommendation_df.sort_values(ascending=False, inplace=True)

    

    # Take first n recommendations from content-based recommender

    movies_copy = movies_df.copy(deep=True)

    movies_copy.set_index('movieId', drop=True, inplace=True)

    top_n_index = recommendation_df.index[:n].tolist()

    results = movies_copy.loc[top_n_index, :]

    results['weighted_average'] = recommendation_df[:n]

    return results
content_based_recommender(7, 10)
def hybrid_recommender(uid, n):

    # Build user profile

    user_movie_ratings = get_user_movie_ratings(uid)

    user_genres_df = get_user_genres(uid)

    

    # Get content-based recommendations (weighted average of genres)

    user_profile = user_genres_df.T.dot(user_movie_ratings['rating'])

    genres_df = movies_with_genres.copy(deep=True).set_index(movies_with_genres['movieId']).drop(['movieId', 'title', 'year'], axis=1)

    recommendation_df = (genres_df.dot(user_profile)) / user_profile.sum()

    recommendation_df.sort_values(ascending=False, inplace=True)

    

    # Take first 100 recommendations from content-based recommender and sort by collaborative score/SVD

    movies_copy = movies_df.copy(deep=True)

    movies_copy.set_index('movieId', drop=True, inplace=True)

    top_n_index = recommendation_df.index[:100].tolist()

    results = movies_copy.loc[top_n_index, :]

    results['weighted_average'] = recommendation_df[:100]

    results['est'] = [svd.predict(uid, x).est for x in results.index.tolist()]

    results.sort_values('est', ascending=False, inplace=True)

    return results.head(n)
hybrid_recommender(7, 10)