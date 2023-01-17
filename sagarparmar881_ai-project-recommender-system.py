# -*- coding: utf-8 -*-
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

#movies = 'drive/My Drive/movies_dataset/movies.csv'
#ratings = 'drive/My Drive/movies_dataset/ratings.csv'

#We are using small dataset because of CRASHING
#movies = '../input/movielens100k/movies.csv'
#ratings = '../input/movielens100k/ratings.csv'

movies = '../input/the-indian-movie-database/collaborative/titles.csv'
ratings = '../input/the-indian-movie-database/collaborative/ratings.csv'


df_movies = pd.read_csv(movies, usecols=['movie_id','title'], dtype={'movie_id':'int32','title':'str'})
df_ratings = pd.read_csv(ratings, usecols=['user_id','movie_id','rating'], dtype={'user_id':'int32','movie_id':'int32','rating':'float32'})

#Count all entries of movies and ratings - 800 Entries
#df_movies.describe()
#df_ratings.describe()

#Spare Matrix
#         Users
#        [4,4,5] A
#Movies  [3,3,4] B ==   Cos(A,B) => 0.95 
#        [3,2,1]

movies_users=df_ratings.pivot(index='movie_id', columns='user_id',values='rating').fillna(0)

mat_movies_users=csr_matrix(movies_users.values)
#movies_users

# Euclidean Distance
# Manhattan Distance
# Minkowski Distance 
# Cosine Similarity
model_knn= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)

def recommender(movie_name, data,model, n_recommendations ):
    model.fit(data)
    idx=process.extractOne(movie_name, df_movies['title'])[2]
    print('Movie Selected: ',df_movies['title'][idx], 'Index: ',idx)
    print('Searching for recommendations.....')
    distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations)
    for i in indices:
        print(df_movies['title'][i].where(i!=idx))
        
entered_movie = input ("Enter a movie name :") 
    
recommender(entered_movie, mat_movies_users, model_knn,20) 