# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Libraries for Recommendation System

from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_movie = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")

data_rating = pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")
movie = data_movie.loc[:,{"movieId","title"}]

rating = data_rating.loc[:,{"userId","movieId","rating"}]
data = pd.merge(movie,rating)

data = data.iloc[:1000000,:]

user_movie_table = data.pivot_table(index = ["title"],columns = ["userId"],values = "rating").fillna(0)

user_movie_table.head(10)
# We choose random movie.

query_index = np.random.choice(user_movie_table.shape[0])

print("Choosen Movie is: ",user_movie_table.index[query_index])
user_movie_table_matrix = csr_matrix(user_movie_table.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')

model_knn.fit(user_movie_table_matrix)

distances, indices = model_knn.kneighbors(user_movie_table.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 6)
movie = []

distance = []



for i in range(0, len(distances.flatten())):

    if i != 0:

        movie.append(user_movie_table.index[indices.flatten()[i]])

        distance.append(distances.flatten()[i])    



m=pd.Series(movie,name='movie')

d=pd.Series(distance,name='distance')

recommend = pd.concat([m,d], axis=1)

recommend = recommend.sort_values('distance',ascending=False)



print('Recommendations for {0}:\n'.format(user_movie_table.index[query_index]))

for i in range(0,recommend.shape[0]):

    print('{0}: {1}, with distance of {2}'.format(i, recommend["movie"].iloc[i], recommend["distance"].iloc[i]))