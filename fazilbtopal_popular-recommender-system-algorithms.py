import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import math

%matplotlib inline



import os

print(os.listdir("../input"))
#Storing the movie information into a pandas dataframe

movies = pd.read_csv('../input/movie.csv')

movies.head()
# Using regular expressions to find a year stored between parentheses

# We specify the parantheses so we don't conflict with movies that have years in their titles

movies['year'] = (movies.title.str.extract('(\(\d\d\d\d\))', expand=False)

                              .str.extract('(\d\d\d\d)', expand=False))  # Removing the parentheses



# Removing the years from the 'title' column

# Strip function to get rid of any ending whitespace characters that may have appeared

movies['title'] = (movies.title.str.replace('(\(\d\d\d\d\))', '')

                               .apply(lambda x: x.strip()))



# Every genre is separated by a | so we simply have to call the split function on |

movies['genres'] = movies.genres.str.split('|')

movies.head()
movies.info() # Check for null elements
# Storing the user information into a pandas dataframe

ratings = pd.read_csv('../input/rating.csv', usecols=['userId', 'movieId', 'rating'],

                     dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32})

ratings.head()
ratings['rating'] = ratings['rating'] * 2

ratings['rating'] = ratings['rating'].astype(np.int8)

ratings.info()
ratings.head()
most_voted = (ratings.groupby('movieId')[['rating']]

                     .count()

                     .sort_values('rating', ascending=False)

                     .reset_index())

most_voted = pd.merge(most_voted, movies, on='movieId').drop('rating', axis=1)

most_voted.head()
# Due to problems with pandas, we can't use pivot_table with our all data as it throws MemoryError.

# Therefore, for this part we will work with a sample data

sample_ratings = ratings.sample(n=100000, random_state=20)



# Creating our sparse matrix and fill NA's with 0 to avoid high memory usage.

pivot = pd.pivot_table(sample_ratings, values='rating', index='userId', columns='movieId', fill_value=0)

pivot.head()
pivot = pivot.astype(np.int8)

pivot.info()
# Let's look something similar to Pulp Fiction

rand_movie = 296



similar = pivot.corrwith(pivot[rand_movie], drop=True).to_frame(name='PearsonR')
rating_count = (ratings.groupby('movieId')[['rating']]

                       .count()

                       .sort_values('rating', ascending=False)

                       .reset_index())

rating_count = pd.merge(rating_count, movies, on='movieId')

rating_count.head()
similar_sum = similar.join(rating_count['rating'])

similar_top10 = similar_sum[similar_sum['rating']>=500].sort_values(['PearsonR', 'rating'], 

                                                            ascending=[False, False]).head(11)

# Add movie names

similar_top10 = pd.merge(similar_top10[1:11], movies[['title', 'movieId']], on='movieId')

similar_top10
from sklearn.decomposition import TruncatedSVD



X = pivot.T

SVD = TruncatedSVD(n_components=500, random_state=20)

SVD_matrix = SVD.fit_transform(X)
SVD.explained_variance_ratio_.sum()
# We'll calculate the Pearson r correlation coefficient, 

# for every movie pair in the resultant matrix. With correlation being 

# based on similarities between user preferences.



corr_mat = np.corrcoef(SVD_matrix)

corr_mat.shape
corr_pulp_fiction = corr_mat[rand_movie]



# Recommending a Highly Correlated Movie.

# We will get different results due to decompression with svd

idx = X[(corr_pulp_fiction < 1.0) & (corr_pulp_fiction > 0.5)].index

movies.loc[idx+1, 'title']