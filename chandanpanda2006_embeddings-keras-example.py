# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import division,print_function

import math, os, json, sys, re

from glob import glob

import numpy as np

from matplotlib import pyplot as plt

import pandas as pd

import PIL

from PIL import Image



import keras

from keras import backend as K

from keras.utils.data_utils import get_file

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Reshape, concatenate, dot, add

from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.regularizers import l2

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD, RMSprop, Adam

from keras.metrics import categorical_crossentropy, categorical_accuracy

from keras.layers.convolutional import *

from keras.preprocessing import image, sequence

from keras.preprocessing.text import Tokenizer



from operator import itemgetter



np.set_printoptions(precision=4, linewidth=100)
%matplotlib inline

%pwd

import os

print(os.listdir("../input"))
path = "../input/"

model_path = '../models/'

if not os.path.exists(model_path): os.mkdir(model_path)

batch_size=64
ratings = pd.read_csv(path+'rating.csv')

ratings.head()
len(ratings)
movies = pd.read_csv(path+'movie.csv')

movies.head()
movie_names = movies.set_index('movieId')['title'].to_dict()
users = ratings.userId.unique()

movies = ratings.movieId.unique()
userid2idx = {o:i for i,o in enumerate(users)}

movieid2idx = {o:i for i,o in enumerate(movies)}
ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])

ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])
user_min, user_max, movie_min, movie_max = (ratings.userId.min(), 

    ratings.userId.max(), ratings.movieId.min(), ratings.movieId.max())

user_min, user_max, movie_min, movie_max
n_users = ratings.userId.nunique()

n_movies = ratings.movieId.nunique()

n_users, n_movies
n_factors = 50

np.random.seed = 42
g=ratings.groupby('userId')['rating'].count()

topUsers=g.sort_values(ascending=False)[:15]



g=ratings.groupby('movieId')['rating'].count()

topMovies=g.sort_values(ascending=False)[:15]



top_r = ratings.join(topUsers, rsuffix='_r', how='inner', on='userId')

top_r = top_r.join(topMovies, rsuffix='_r', how='inner', on='movieId')



pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)

msk = np.random.rand(len(ratings)) < 0.05

trn = ratings[msk]

msk = np.random.rand(len(ratings)) < 0.99

val = ratings[~msk]
userid2idx = {o:i for i,o in enumerate(users)}

movieid2idx = {o:i for i,o in enumerate(movies)}
user_in = Input(shape=(1,), dtype='int64', name='user_in')

u = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=l2(1e-4))(user_in)

movie_in = Input(shape=(1,), dtype='int64', name='movie_in')

m = Embedding(n_movies, n_factors, input_length=1, embeddings_regularizer=l2(1e-4))(movie_in)
x = dot([u, m], axes=-1, normalize=False)

x = Flatten()(x)

model = Model([user_in, movie_in], x)

model.compile(Adam(0.001), loss='mse')

model.summary()
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, epochs=1, 

          validation_data=([val.userId, val.movieId], val.rating))
model.optimizer.lr=0.01
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, epochs=3, 

          validation_data=([val.userId, val.movieId], val.rating))
model.optimizer.lr=0.001
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, epochs=6, 

          validation_data=([val.userId, val.movieId], val.rating))
def embedding_input(name, n_in, n_out, reg):

    inp = Input(shape=(1,), dtype='int64', name=name)

    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)
user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)

movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)
def create_bias(inp, n_in):

    x = Embedding(n_in, 1, input_length=1)(inp)

    return Flatten()(x)
ub = create_bias(user_in, n_users)

mb = create_bias(movie_in, n_movies)
x = dot([u, m], axes=-1)

x = Flatten()(x)

x = add([x, ub])

x = add([x, mb])

model = Model([user_in, movie_in], x)

model.compile(Adam(0.001), loss='mse')

model.summary()
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, epochs=1, 

          validation_data=([val.userId, val.movieId], val.rating))
model.optimizer.lr=0.01
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, epochs=6, 

          validation_data=([val.userId, val.movieId], val.rating))
model.optimizer.lr=0.001
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, epochs=10, 

          validation_data=([val.userId, val.movieId], val.rating))
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=5, 

          validation_data=([val.userId, val.movieId], val.rating))
model.predict([np.array([3]), np.array([6])])
g=ratings.groupby('movieId')['rating'].count()

topMovies=g.sort_values(ascending=False)[:2000]

topMovies = np.array(topMovies.index)
get_movie_bias = Model(movie_in, mb)

movie_bias = get_movie_bias.predict(topMovies)

movie_ratings = [(b[0], movie_names[movies[i]]) for i,b in zip(topMovies,movie_bias)]
sorted(movie_ratings, key=itemgetter(0))[:15]
sorted(movie_ratings, key=itemgetter(0), reverse=True)[:15]
get_movie_emb = Model(movie_in, m)

movie_emb = np.squeeze(get_movie_emb.predict([topMovies]))

movie_emb.shape
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

movie_pca = pca.fit(movie_emb.T).components_
fac0 = movie_pca[0]
movie_comp = [(f, movie_names[movies[i]]) for f,i in zip(fac0, topMovies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
fac1 = movie_pca[1]
movie_comp = [(f, movie_names[movies[i]]) for f,i in zip(fac1, topMovies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
fac2 = movie_pca[2]
movie_comp = [(f, movie_names[movies[i]]) for f,i in zip(fac2, topMovies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
import sys

stdout, stderr = sys.stdout, sys.stderr # save notebook stdout and stderr

sys.stdout, sys.stderr = stdout, stderr # restore notebook stdout and stderr
start=50; end=100

X = fac0[start:end]

Y = fac2[start:end]

plt.figure(figsize=(15,15))

plt.scatter(X, Y)

for i, x, y in zip(topMovies[start:end], X, Y):

    plt.text(x,y,movie_names[movies[i]], color=np.random.rand(3)*0.7, fontsize=14)

plt.show()
user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)

movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)
x = concatenate([u, m])

x = Flatten()(x)

x = Dropout(0.3)(x)

x = Dense(70, activation='relu')(x)

x = Dropout(0.75)(x)

x = Dense(1)(x)

nn = Model([user_in, movie_in], x)

nn.compile(Adam(0.001), loss='mse')
nn.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, epochs=8, 

          validation_data=([val.userId, val.movieId], val.rating))