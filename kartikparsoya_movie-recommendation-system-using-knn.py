import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

from fuzzywuzzy import process
movie=pd.read_csv('../input/movielens-20m-dataset/movie.csv')

rating=pd.read_csv('../input/movielens-20m-dataset/rating.csv')

tag=pd.read_csv('../input/movielens-20m-dataset/tag.csv')
movie.head()
movie['year'] = movie.title.str.extract('(\(\d\d\d\d\))',expand=False)

#Removing the parentheses

movie['year'] = movie.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column

movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')

#Applying the strip function to get rid of any ending whitespace characters that may have appeared

movie['title'] = movie['title'].str.strip()

movie.head()
rating.groupby('movieId').rating.mean()

rating.head()
df=movie.join(rating,lsuffix='N', rsuffix='K')

df
df=df.drop(['movieIdK','genres','year','timestamp'],axis=1)
# There will be lot of nan value in our new dataframe as it is highly unlikely that that ours users have seen almost all movies.

movie_users=df.pivot(index='movieIdN', columns='userId',values='rating').fillna(0)

matrix_movies_users=csr_matrix(movie_users.values)

print(matrix_movies_users)
knn= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
knn.fit(matrix_movies_users)
def recommender(movie_name, data,model, n_recommendations ):

    model.fit(data)

    idx=process.extractOne(movie_name, df['title'])[2]

    print('Movie Selected:-',df['title'][idx], 'Index: ',idx)

    print('Searching for recommendations.....')

    distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations)

    for i in indices:

        print(df['title'][i].where(i!=idx))

    

recommender('jumanji', matrix_movies_users, knn,5)