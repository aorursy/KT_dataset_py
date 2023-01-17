# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.user', sep='|', names=users_cols, parse_dates=True) 

users.head()
rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.data', sep='\t', names=rating_cols)
ratings.head()
movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.item', sep='|', names=movie_cols, usecols=range(5),encoding='latin-1')
movies.head()
ratings=ratings.loc[:,ratings.columns !='unix_timestamp']
movies=movies.loc[:,['movie_id','title']]
data=pd.merge(ratings,movies)
data.head()
data['user_id'].nunique()
data['movie_id'].nunique()
user_movie_table=data.pivot_table(index=['title'],columns=['user_id'],values='rating').fillna(0)
user_movie_table.head(10)
#choose a random movie
query_index=np.random.choice(user_movie_table.shape[0])
print('The movie that is chosen is : ',user_movie_table.index[query_index])
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

user_movie_table_matrix=csr_matrix(user_movie_table.values)

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










