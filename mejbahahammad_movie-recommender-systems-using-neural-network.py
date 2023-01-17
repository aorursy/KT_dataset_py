import numpy as np

import pandas as pd

import os, time, re

import pickle, gzip, datetime



from datetime import datetime



now = datetime.now()



import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

import matplotlib as mpl

%matplotlib inline





from sklearn import preprocessing as pp

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error

import lightgbm as lgb





import tensorflow as tf

import keras



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Activation, Dense, Dropout

from tensorflow.keras.layers import BatchNormalization, Input, Lambda

from tensorflow.keras.layers import Embedding, Flatten, dot

from tensorflow.keras import regularizers

from tensorflow.keras.losses import mse, binary_crossentropy
ratingDF = pd.read_csv('../input/movielens-20m-dataset/ratings.csv')
ratingDF.userId = ratingDF.userId.astype(str).astype(int)

ratingDF.movieId = ratingDF.movieId.astype(str).astype(int)

ratingDF.rating = ratingDF.rating.astype(str).astype(float)

ratingDF.timestamp = ratingDF.timestamp.apply(lambda x: now.strftime("%m/%d/%Y, %H:%M:%S"))
n_users = ratingDF.userId.unique().shape[0]

n_movies = ratingDF.movieId.unique().shape[0]

n_ratings = len(ratingDF)

avg_ratings_per_user = n_ratings/n_users
print('Number of unique users: ', n_users)

print('Number of unique movies: ', n_movies)

print('Number of total ratings: ', n_ratings)

print('Average number of ratings per user: ', avg_ratings_per_user)
movieIndex = ratingDF.groupby("movieId").count().sort_values(by= \

"rating",ascending=False)[0:1000].index

ratingDFX2 = ratingDF[ratingDF.movieId.isin(movieIndex)]

ratingDFX2.count()
userIndex = ratingDFX2.groupby("userId").count().sort_values(by= \

"rating",ascending=False).sample(n=1000, random_state=2018).index

ratingDFX3 = ratingDFX2[ratingDFX2.userId.isin(userIndex)]

ratingDFX3.count()
movies = ratingDFX3.movieId.unique()

moviesDF = pd.DataFrame(data=movies,columns=['originalMovieId'])

moviesDF['newMovieId'] = moviesDF.index+1
users = ratingDFX3.userId.unique()

usersDF = pd.DataFrame(data=users,columns=['originalUserId'])

usersDF['newUserId'] = usersDF.index+1
ratingDFX3 = ratingDFX3.merge(moviesDF,left_on='movieId', \

right_on='originalMovieId')

ratingDFX3.drop(labels='originalMovieId', axis=1, inplace=True)

ratingDFX3 = ratingDFX3.merge(usersDF,left_on='userId', \

right_on='originalUserId')

ratingDFX3.drop(labels='originalUserId', axis=1, inplace=True)
n_users = ratingDFX3.userId.unique().shape[0]

n_movies = ratingDFX3.movieId.unique().shape[0]

n_ratings = len(ratingDFX3)

avg_ratings_per_user = n_ratings/n_users
print('Number of unique users: ', n_users)

print('Number of unique movies: ', n_movies)

print('Number of total ratings: ', n_ratings)

print('Average number of ratings per user: ', avg_ratings_per_user)
X_train, X_test = train_test_split(ratingDFX3,

test_size=0.10, shuffle=True, random_state=2018)

X_validation, X_test = train_test_split(X_test,

test_size=0.50, shuffle=True, random_state=2018)
print('Shape of train set:', X_train.shape)

print('Shape of validation set:',X_validation.shape)

print('Shape of test set: ',X_test.shape)
print('Size of train set:', X_train.size)

print('Size of validation set:',X_validation.size)

print('Size of test set: ',X_test.size)
# Generate ratings matrix for train

ratings_train = np.zeros((n_users, n_movies))

for row in X_train.itertuples():

    ratings_train[row[6]-1, row[5]-1] = row[3]
sparsity = float(len(ratings_train.nonzero()[0]))

sparsity /= (ratings_train.shape[0] * ratings_train.shape[1])

sparsity *= 100

print('Sparsity: {:4.2f}%'.format(sparsity))
# Generate ratings matrix for validation

ratings_validation = np.zeros((n_users, n_movies))

for row in X_validation.itertuples():

    ratings_validation[row[6]-1, row[5]-1] = row[3]
# Generate ratings matrix for test

ratings_test = np.zeros((n_users, n_movies))

for row in X_test.itertuples():

    ratings_test[row[6]-1, row[5]-1] = row[3]
actual_validation = ratings_validation[ratings_validation.nonzero()].flatten()
pred_validation = np.zeros((len(X_validation),1))

pred_validation[pred_validation==0] = 3.5

pred_validation
print("Mean  Squared Error is : ", mean_squared_error(pred_validation, actual_validation))
ratings_validation_prediction = np.zeros((n_users, n_movies))

i = 0

for row in ratings_train:

    ratings_validation_prediction[i][ratings_validation_prediction[i]==0] = np.mean(row[row>0])

    i += 1
pred_validation = ratings_validation_prediction  [ratings_validation.nonzero()].flatten()

user_average = mean_squared_error(pred_validation, actual_validation)

print('Mean squared error using user average:', user_average)
ratings_validation_prediction = np.zeros((n_users, n_movies)).T

i = 0

for row in ratings_train.T:

    ratings_validation_prediction[i][ratings_validation_prediction[i]==0] = np.mean(row[row>0])

    i += 1
ratings_validation_prediction = ratings_validation_prediction.T

pred_validation = ratings_validation_prediction[ratings_validation.nonzero()].flatten()

movie_average = mean_squared_error(pred_validation, actual_validation)

print('Mean squared error using movie average:', movie_average)
n_latent_factors = 1

user_input = Input(shape=[1], name='user')

user_embedding = Embedding(input_dim=n_users + 1, output_dim=n_latent_factors,

name='user_embedding')(user_input)

user_vec = Flatten(name='flatten_users')(user_embedding)

movie_input = Input(shape=[1], name='movie')

movie_embedding = Embedding(input_dim=n_movies + 1,

output_dim=n_latent_factors,

name='movie_embedding')(movie_input)

movie_vec = Flatten(name='flatten_movies')(movie_embedding)

product = dot([movie_vec, user_vec], axes=1)

model = Model(inputs=[user_input, movie_input], outputs=product)

model.compile('adam', 'mean_squared_error')
history = model.fit(x=[X_train.newUserId, X_train.newMovieId],

                    y=X_train.rating, epochs=100,

                    validation_data=([X_validation.newUserId,

                                      X_validation.newMovieId], X_validation.rating),

                    verbose=1)
pd.Series(history.history['val_loss'][10:]).plot(logy=False)

plt.xlabel("Epoch")

plt.ylabel("Validation Error")

print('Minimum MSE: ', min(history.history['val_loss']))