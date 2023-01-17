# Reference: https://github.com/skrinak/recommendationEngine

# Import needed libraries

import os

import urllib.request

import zipfile

import logging



import mxnet as mx

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



# Some configurations

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

sns.set_style('whitegrid')

%matplotlib inline
# Fetch the movielens dataset

%time

if not os.path.exists('ml-20m.zip'):

    urllib.request.urlretrieve('http://files.grouplens.org/datasets/movielens/ml-20m.zip', 'ml-20m.zip')

with zipfile.ZipFile("ml-20m.zip", "r") as f:

    f.extractall("./")



# Read the data. I have limited already the 

# columns that I will use as I have scanned it already before

data = pd.read_csv('./ml-20m/ratings.csv', sep=',', usecols=(0, 1, 2))

data.describe()
data.sample(3)
# Describe the data a bit. I would like to know 

# the distribution (in terms of count) of ratings across the dataset

plt.hist(data['rating'])

plt.xlabel("Rating", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.title("Distribution of Ratings in MovieLens 20M", fontsize=14)

plt.show()
# As you could see from the data that we have, we have users, movies and ratings

# I would like to know how many are unique users and how many are unique movie

print("user id min/max: ", data['userId'].min(), data['userId'].max())

print("# unique users: ", np.unique(data['userId']).shape[0])

print("")

print("movie id min/max: ", data['movieId'].min(), data['movieId'].max())

print("# unique movies: ", np.unique(data['movieId']).shape[0])
# Assigning this figures to a variable 

n_users, n_movies = 138493, 131262

batch_size = 25000
# Reshuffle the indexes

data = data.sample(frac=1).reset_index(drop=True)

n = 19000000



# Get the first 19000000 index as train data

train_users   = data['userId'].values[:n] - 1 # Offset by 1

train_movies  = data['movieId'].values[:n] - 1 # Offset by 1 

train_ratings = data['rating'].values[:n]



# Get the remaining of the 19000000 index as validation data

valid_users   = data['userId'].values[n:] - 1 # Offset by 1

valid_movies  = data['movieId'].values[n:] - 1 # Offset by 1

valid_ratings = data['rating'].values[n:]
## Train the mxnet model. I will use nn for neural net

## I will use adam as optimizer. Of course you could argue why but 

## this is the most common approach

#X_train = mx.io.NDArrayIter({'user': train_users, 'movie': train_movies}, 

#                            label=train_ratings, batch_size=batch_size)

#X_val   = mx.io.NDArrayIter({'user': valid_users, 'movie': valid_movies}, 

#                            label=valid_ratings, batch_size=batch_size)

#

## I've reduced the output dimension of users to 15 since

## we found that there are extermely more number of unique users compared to unique movies

## Time to make the embeddings

#user  = mx.symbol.Variable("user")

#user  = mx.symbol.Embedding(data=user, input_dim=n_users, output_dim=15)

#movie = mx.symbol.Variable("movie")

#movie = mx.symbol.Embedding(data=movie, input_dim=n_movies, output_dim=25)

#

#y_true = mx.symbol.Variable("softmax_label")

#

## Forming the neural network, we also need to flatten the data as usual

# We will be putting two dense layers here

#nn = mx.symbol.concat(user, movie)

#nn = mx.symbol.flatten(nn)

#nn = mx.symbol.FullyConnected(data=nn, num_hidden=64)

#nn = mx.symbol.Activation(data=nn, act_type='relu') 

#nn = mx.symbol.FullyConnected(data=nn, num_hidden=64)

#nn = mx.symbol.Activation(data=nn, act_type='relu')

#nn = mx.symbol.FullyConnected(data=nn, num_hidden=1)

#

#y_pred = mx.symbol.LinearRegressionOutput(data=nn, label=y_true)

#

## Train the model (neural network)

#model = mx.module.Module(context=mx.cpu(), data_names=('user', 'movie'), symbol=y_pred)

#model.fit(X_train, num_epoch=5, optimizer='adam', optimizer_params=(('learning_rate', 0.05),),

#          eval_metric='rmse', eval_data=X_val, batch_end_callback=mx.callback.Speedometer(batch_size, 250))