# to open files

import pandas as pd



# for numerical operations

import numpy as np



# sci-kit learn to measure distances

from sklearn.metrics.pairwise import pairwise_distances
header = ['user_id', 'item_id', 'rating', 'timestamp']

movielens_data = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data', sep='\t', names=header)

movielens_data.head()
movielens_data.shape
n_users, n_movies  = movielens_data['user_id'].nunique(), movielens_data['item_id'].nunique()

n_users, n_movies
# We can also use panda's pivot_table to create this 2D matrix. But I'll keep it simple by doing it mannually.

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html



train_data_matrix = np.zeros((n_users, n_movies))



for line in movielens_data.itertuples():

    train_data_matrix[line[1]-1, line[2]-1] = line[3]

    

train_data_matrix.shape
train_data_matrix
user_distances = pairwise_distances(train_data_matrix, metric="cosine")



# ".T" below is to transpose our 2D matrix.

train_data_matrix_transpose = train_data_matrix.T

movie_distances = pairwise_distances(train_data_matrix_transpose, metric="cosine")



user_distances.shape, movie_distances.shape
user_distances
movie_distances
user_similarity = 1 - user_distances

movie_similarity = 1 - movie_distances
user_similarity
movie_similarity
idx_to_movie = {}



with open('../input/movielens-100k-dataset/ml-100k/u.item', 'r', encoding="ISO-8859-1") as f:

    for line in f.readlines():

        info = line.split('|')

        idx_to_movie[int(info[0])-1] = info[1]



movie_to_idx = {v: k for k, v in idx_to_movie.items()}
idx_to_movie[0], idx_to_movie[1], idx_to_movie[2], idx_to_movie[3] 
movie_to_idx['Toy Story (1995)'], movie_to_idx['GoldenEye (1995)'], movie_to_idx['Four Rooms (1995)'], movie_to_idx['Get Shorty (1995)'] 
# What we do is, we just that movie's column & sort it by value.

# Those value represents "similarity" so, we just need to sort it & pick first "k" values.



def top_k_movies(similarity, mapper, movie_idx, k=6):

    return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-2:-1]]
favorite_movie_name = 'Batman Forever (1995)'

movie_index = movie_to_idx[favorite_movie_name]

movie_index
how_much_movie_to_show = 7



movies = top_k_movies(movie_similarity, idx_to_movie, movie_index, k = how_much_movie_to_show)

movies[1:how_much_movie_to_show + 1]
favorite_movie_name = 'Star Wars (1977)'

movie_index = movie_to_idx[favorite_movie_name]

movie_index
how_much_movie_to_show = 7



movies = top_k_movies(movie_similarity, idx_to_movie, movie_index, k = how_much_movie_to_show)

movies[1:how_much_movie_to_show + 1]