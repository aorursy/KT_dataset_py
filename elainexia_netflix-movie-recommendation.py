# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# this is just to know how much time will it take to run this entire ipython notebook 
from datetime import datetime
# globalstart = datetime.now()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('nbagg')

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import seaborn as sns
sns.set_style('whitegrid')
import os
from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random
start = datetime.now()

# if os.path.isfile('data.csv'):
#     os.remove('data.csv')

if not os.path.isfile('data1.csv'):
    # Create a file 'data.csv' before reading it
    # Read all the files in netflix and store them in one big file('data.csv')
    # We re reading from each of the four files and appendig each rating to a global file 'train.csv'
    data = open('data1.csv', mode='w')
    
    row = list()
    files=['../input/netflix-prize-data/combined_data_1.txt','../input/netflix-prize-data/combined_data_2.txt', 
           '../input/netflix-prize-data/combined_data_3.txt', '../input/netflix-prize-data/combined_data_4.txt']
    for file in files:
        print("Reading ratings from {}...".format(file))
        with open(file) as f:
            for line in f: 
                del row[:] # you don't have to do this.
                line = line.strip()
                if line.endswith(':'):
                    # All below are ratings for this movie, until another movie appears.
                    movie_id = line.replace(':', '')
                else:
                    row = [x for x in line.split(',')]
                    row.insert(0, movie_id)
                    data.write(','.join(row))
                    data.write('\n')
        print("Done.\n")
    data.close()
print('Time taken :', datetime.now() - start)
print("creating the dataframe from data1.csv file..")
ratings = pd.read_csv('data1.csv', sep=',', 
                       names=['movie', 'user','rating','date'])
ratings.date = pd.to_datetime(ratings.date)
print('Done.\n')

# we are arranging the ratings according to time.
print('Sorting the dataframe by date..')
ratings.sort_values(by='date', inplace=True)
print('Done..')
# Rename the column names
ratings.rename(columns={'movie':'movieId',
                          'user':'userId',
                          'date':'timestamp'}, 
                 inplace=True)
# Sample top 1K users w/ top 10K movies
g = ratings.groupby('userId')['rating'].count()
top_users = g.sort_values(ascending=False)[:1000]
g = ratings.groupby('movieId')['rating'].count()
top_movies = g.sort_values(ascending=False)[:10000]
top_r = ratings.join(top_users, rsuffix='_r', how='inner', on='userId')
ratings_sample = top_r.join(top_movies, rsuffix='_r', how='inner', on='movieId')
ratings_sample.shape
ratings.shape
ratings_sample.head()
ratings_sample
# rating_df[rating_df['user']==510180].sort_values(by='movie')
imdb_file = '../input/imdb-extensive-dataset/IMDb movies.csv'

imdb_df = pd.read_csv(imdb_file)

imdb_df.head()
title_df = pd.read_csv('../input/netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['movie', 'year', 'title'])
title_df.head(10)
title_df.rename(columns={'movie':'movieId'}, inplace=True)
# Netflix user to movie rating data
rating_df.head()
# Netflix movie ID to title data
title_df.head()
# IMDB movie metadata
imdb_df.head()
rating_df.merge(title_df, left_on='movie', right_on='movie').head()
title_df.head()
imdb_df.head()
# Overlap 
pd.merge(title_df, imdb_df, how = 'inner', left_on = 'title', right_on = 'title').shape
title_df.shape
g = ratings_sample.groupby('userId')['rating'].count()
top_users = g.sort_values(ascending=False)[:15]
g = ratings_sample.groupby('movieId')['rating'].count()
top_movies = g.sort_values(ascending=False)[:15]
top_r = ratings_sample.join(top_users, rsuffix='_r', how='inner', on='userId')
top_r = top_r.join(top_movies, rsuffix='_r', how='inner', on='movieId')
pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)
# Encode the user and movie from 0 to distinct number of users / movies
user_enc = LabelEncoder()
ratings_sample['user_enc'] = user_enc.fit_transform(ratings_sample['userId'].values)
n_users = ratings_sample['user_enc'].nunique()
item_enc = LabelEncoder()
ratings_sample['movie_enc'] = item_enc.fit_transform(ratings_sample['movieId'].values)
n_movies = ratings_sample['movie_enc'].nunique()
ratings_sample['rating'] = ratings_sample['rating'].values.astype(np.float32)
min_rating = min(ratings_sample['rating'])
max_rating = max(ratings_sample['rating'])
n_users, n_movies, min_rating, max_rating
X = ratings_sample[['user_enc', 'movie_enc']].values
y = ratings_sample['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
n_factors = 50
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]
from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
def RecommenderV1(n_users, n_movies, n_factors):
    user = Input(shape=(1,))
    u = Embedding(n_users, n_factors, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(user)
    u = Reshape((n_factors,))(u)
    
    movie = Input(shape=(1,))
    m = Embedding(n_movies, n_factors, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(movie)
    m = Reshape((n_factors,))(m)
    
    x = Dot(axes=1)([u, m])
    model = Model(inputs=[user, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model
model = RecommenderV1(n_users, n_movies, n_factors)
model.summary()
history = model.fit(x=X_train_array, y=y_train, batch_size=64, epochs=5,
                    verbose=1, validation_data=(X_test_array, y_test))
from keras.layers import Add, Activation, Lambda
class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x
def RecommenderV2(n_users, n_movies, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    ub = EmbeddingLayer(n_users, 1)(user)
    
    movie = Input(shape=(1,))
    m = EmbeddingLayer(n_movies, n_factors)(movie)
    mb = EmbeddingLayer(n_movies, 1)(movie)
    x = Dot(axes=1)([u, m])
    x = Add()([x, ub, mb])
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
    model = Model(inputs=[user, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model
model2 = RecommenderV2(n_users, n_movies, n_factors, min_rating, max_rating)
model2.summary()
history2 = model2.fit(x=X_train_array, y=y_train, batch_size=64, epochs=5,
                    verbose=1, validation_data=(X_test_array, y_test))
from keras.layers import Concatenate, Dense, Dropout
def RecommenderNet(n_users, n_movies, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    
    movie = Input(shape=(1,))
    m = EmbeddingLayer(n_movies, n_factors)(movie)
    
    x = Concatenate()([u, m])
    x = Dropout(0.05)(x)
    
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
    model = Model(inputs=[user, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model
model3 = RecommenderNet(n_users, n_movies, n_factors, min_rating, max_rating)
model3.summary()
history = model3.fit(x=X_train_array, y=y_train, batch_size=64, epochs=5,
                    verbose=1, validation_data=(X_test_array, y_test))
# Get Embeddings for Movies
movie_layer = model.get_layer('embedding_1')
movie_weights = movie_layer.get_weights()[0]
# movie_weights[:10]
movie_weights_df = pd.DataFrame(movie_weights).reset_index()
movie_weights_df.rename(columns={'index':'movie_enc'}, inplace=True)
movie_weights_df.head()
# Get Movie to Titles Mapping
movie_list = ratings_sample[['movie_enc','movieId']].drop_duplicates()
movie_titles = movie_list.merge(title_df, left_on='movieId', right_on='movieId')
# movie_titles.head()
# Append Titles to Embedding
movie_weights_titles_df = movie_weights_df.merge(movie_titles, left_on = 'movie_enc', right_on = 'movie_enc')
movie_weights_titles_df.head()
# The top 15 movies watched by Netflix users
top_movies
# Check the specific movie
movie_titles[movie_titles.movieId == 14410]
movie_weights_titles_df[movie_weights_titles_df.movie_enc == 8063]
# Embedding only dataframe
movie_weights_only_df = pd.DataFrame(movie_weights)
# Calculate distances between the specific movie and all other movies
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

# the_movie_embedding_only_normalized = preprocessing.normalize(np.array(the_movie_embedding_only), norm='l2')
movie_weights_only_df.iloc[:,:] = Normalizer(norm='l2').fit_transform(movie_weights_only_df)
the_movie_embedding_only_normalized = movie_weights_only_df[movie_weights_only_df.index == 8063]
the_movie_dot_product_normalizer = movie_weights_only_df.dot(np.transpose(the_movie_embedding_only_normalized))
# Get the distances
the_movie_dot_product_normalizer.sort_values(by = 8063, ascending = False).head(15)
movie_weights_titles_df[movie_weights_titles_df.movie_enc == 9873]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
movie_weights_only_pca = pca.fit_transform(movie_weights_only_df)
pca.explained_variance_ratio_
movie_weights_only_pca_df = pd.DataFrame(movie_weights_only_pca)
movie_weights_only_pca_df = pd.DataFrame(movie_weights_only_pca_df).reset_index()
movie_weights_only_pca_df.rename(columns={'index':'movie_enc'}, inplace=True)
movie_weights_only_pca_df.head()
movie_genre = pd.merge(movie_titles, imdb_df, how = 'inner', left_on = 'title', right_on = 'title')
movie_genre_df = movie_genre[['movie_enc', 'movieId', 'year_x', 'title', 'director', 'genre']]
# movie_genre_df.head()
movie_pca_genre = movie_weights_only_pca_df.merge(movie_genre_df, left_on = 'movie_enc', right_on = 'movie_enc')
movie_pca_genre.rename(columns={0:'pca_0'}, inplace=True)
movie_pca_genre.rename(columns={1:'pca_1'}, inplace=True)
movie_pca_genre.groupby('new_genre').count()['movie_enc'].sort_values(ascending=False)
# movie_pca_genre.genre
new = movie_pca_genre["genre"].str.split(", ", n = 1, expand = True)
movie_pca_genre['new_genre'] = new[0]
movie_pca_genre.head()
sample_data = movie_pca_genre[movie_pca_genre.new_genre.isin(['Horror', 'Action', 'Musical'])]
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
ax = sns.scatterplot(x = "pca_0", y = "pca_1", hue = "new_genre", data = sample_data)

