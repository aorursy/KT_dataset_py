# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
ratings = pd.read_csv("/kaggle/input/toy_dataset.csv", index_col = 0)
ratings.fillna(0, inplace = True)
def standardize(row):
    new_row = (row - row.mean())/ (row.max() - row.min())
    return new_row
ratings_std = ratings.apply(standardize)
ratings_std
item_similarity = cosine_similarity(ratings_std.T)
print(item_similarity)
item_similarity_df = pd.DataFrame(item_similarity, index = ratings.columns, columns = ratings.columns)
item_similarity_df
def get_similar_movies(movie_name, user_rating):
    similar_score = item_similarity_df[movie_name] * (user_rating - 2.5 )
    similar_score = similar_score.sort_values(ascending = False)
    return similar_score

print(get_similar_movies("action3", 5))
action_lover = [("action1", 5), ("romantic2", 1), ("romantic3", 1)]
similar_movies = pd.DataFrame()
for movie, rating in action_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie, rating), ignore_index = True)
similar_movies.head()
similar_movies.sum().sort_values(ascending = False)
ratings = pd.read_csv("/kaggle/input/ratings.csv")
movies = pd.read_csv("/kaggle/input/movies.csv")
ratings = pd.merge(movies, ratings)
ratings.head()
ratings.drop(['genres', 'timestamp'], axis = 1)
user_ratings = ratings.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
user_ratings.head()
user_ratings = user_ratings.dropna(thresh = 10, axis = 1).fillna(0, axis = 1)
print(user_ratings)
corrMatrix = user_ratings.corr(method='pearson')
corrMatrix.head(100)
def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating - 2.5)
    similar_ratings = similar_ratings.sort_values(ascending = False)
    #print(type(similar_ratings))
    return similar_ratings
romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]
similar_movies = pd.DataFrame()
for movie,rating in romantic_lover:
    similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

similar_movies.head(10)
action_lover = [("Amazing Spider-Man, The (2012)",5),("Mission: Impossible III (2006)",4),("Toy Story 3 (2010)",2),("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",4)]
similar_movies = pd.DataFrame()
for movie,rating in action_lover:
    similar_movies = similar_movies.append(get_similar(movie,rating), ignore_index = True)

similar_movies.head(10)
similar_movies.sum().sort_values(ascending = False).head(20)