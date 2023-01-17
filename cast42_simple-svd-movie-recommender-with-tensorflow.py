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
ratings = pd.read_csv('../input/ratings_small.csv')
ratings.head()
user_item = ratings.groupby(['userId', 'movieId'])['rating'].first().unstack(fill_value=0.0)
user_item.shape
uim = user_item.values
# Based on https://www.bonaccorso.eu/2017/08/02/svd-recommendations-using-tensorflow/
import tensorflow as tf
# Create a Tensorflow graph
graph = tf.Graph()
nb_users = user_item.shape[0]
nb_movies = user_item.shape[1]
nb_factors = 500
max_rating = 5
nb_rated_movies = 5
top_k_movies = 10
with graph.as_default():
    # User-item matrix
    user_item_matrix = tf.placeholder(tf.float32, shape=(nb_users, nb_movies))
    
    # SVD
    St, Ut, Vt = tf.svd(user_item_matrix)
    
    # Compute reduced matrices
    Sk = tf.diag(St)[0:nb_factors, 0:nb_factors]
    Uk = Ut[:, 0:nb_factors]
    Vk = Vt[0:nb_factors, :]
    
    # Compute Su and Si
    Su = tf.matmul(Uk, tf.sqrt(Sk))
    Si = tf.matmul(tf.sqrt(Sk), Vk)
    
    # Compute user ratings
    ratings_t = tf.matmul(Su, Si)
    
    # Pick top k suggestions
    best_ratings_t, best_items_t = tf.nn.top_k(ratings_t, top_k_movies)
# Create Tensorflow session
session = tf.InteractiveSession(graph=graph)
# Compute the top k suggestions for all users
feed_dict = {
    user_item_matrix: uim
}
%time best_items = session.run([best_items_t], feed_dict=feed_dict)
# Suggestions for user 1000, 1010
for i in range(100, 110):
    print('User {}: {}'.format(i, best_items[0][i]))
