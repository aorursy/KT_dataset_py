import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
ratings = pd.read_csv('../input/ratings.csv')
ratings.head(10)
ratings.shape
users = ratings.userId.unique()
movies = ratings.movieId.unique()
n_user = len(users)
n_movie =  len(movies)
print(n_user , n_movie)
ratings["userId"].value_counts().sort_values(ascending=False)[0:15]
ratings["movieId"].value_counts().sort_values(ascending=False)[0:15]
userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}
ratings['userId'] = ratings['userId'].apply(lambda x: userid2idx[x])
ratings['movieId'] = ratings['movieId'].apply(lambda x: movieid2idx[x])
split = np.random.rand(len(ratings)) < 0.8
train = ratings[split]
valid = ratings[~split]
print(train.shape , valid.shape)
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2 , l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy
filepath="recommend_weights.hdf5"
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
factor =30
user_input = Input(shape=(1,), dtype='int64', name='user_input')
user_emb = Embedding(n_user, factor, input_length=1, W_regularizer=l2(1e-4))(user_input)
movie_input = Input(shape=(1,), dtype='int64', name='movie_input')
movie_emb = Embedding(n_movie, factor, input_length=1, W_regularizer=l2(1e-4))(movie_input)
user_bias =  Embedding(n_user, 1, input_length = 1)(user_input)
user_bias = Flatten()(user_bias)
movie_bias = Embedding(n_movie, 1, input_length =1)(movie_input)
movie_bias = Flatten()(movie_bias)
lr =0.0001
# Building a linear model
inp = merge([user_emb, movie_emb], mode = 'dot')
inp = Flatten()(inp)
inp = keras.layers.add(([inp, user_bias]))
inp = Dense(16 , activation = 'relu')(inp)
inp = Dropout(0.4)(inp)
inp = Dense(1)(inp)
inp = keras.layers.add(([inp, movie_bias]))
model = Model([user_input, movie_input], inp)
model.compile(Adam(lr), loss = 'mse')
model.summary()
history = model.fit([train.userId, train.movieId], train.rating, batch_size=64, epochs=15, 
          validation_data=([valid.userId, valid.movieId], 
            valid.rating) , callbacks=[checkpoint])
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(history.history['loss'] , 'g')
plt.plot(history.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
history = model.fit([train.userId, train.movieId], train.rating, batch_size=128, epochs=10, 
          validation_data=([valid.userId, valid.movieId], 
            valid.rating))
factor =64
user_input = Input(shape=(1,), dtype='int64', name='user_input')
user_emb = Embedding(n_user, factor, input_length=1, W_regularizer=l2(1e-4))(user_input)
movie_input = Input(shape=(1,), dtype='int64', name='movie_input')
movie_emb = Embedding(n_movie, factor, input_length=1, W_regularizer=l2(1e-4))(movie_input)
user_bias =  Embedding(n_user, 1, input_length = 1)(user_input)
user_bias = Flatten()(user_bias)
movie_bias = Embedding(n_movie, 1, input_length =1)(movie_input)
movie_bias = Flatten()(movie_bias)
lr =0.0001
# Building a linear model
inp = merge([user_emb, movie_emb], mode = 'dot')
inp = Flatten()(inp)
inp = keras.layers.add(([inp, user_bias]))
inp = Dense(16 , activation = 'relu')(inp)
inp = Dropout(0.4)(inp)
inp = Dense(1)(inp)
inp = keras.layers.add(([inp, movie_bias]))
model = Model([user_input, movie_input], inp)
model.compile(Adam(lr), loss = 'mse')
model.summary()
history = model.fit([train.userId, train.movieId], train.rating, batch_size=64, epochs=15, 
          validation_data=([valid.userId, valid.movieId], 
            valid.rating) , callbacks=[checkpoint])
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(history.history['loss'] , 'p')
plt.plot(history.history['val_loss'] , 'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
model.optimizer.lr = 0.00001
history = model.fit([train.userId, train.movieId], train.rating, batch_size=128, epochs=2, 
          validation_data=([valid.userId, valid.movieId], 
            valid.rating))
model.optimizer.lr = 0.000001
history = model.fit([train.userId, train.movieId], train.rating, batch_size=128, epochs=2, 
          validation_data=([valid.userId, valid.movieId], 
            valid.rating))
