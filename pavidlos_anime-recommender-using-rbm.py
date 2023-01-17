# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
rating_data = pd.read_csv('../input/rating.csv')
anime_data = pd.read_csv('../input/anime.csv')
anime_list = anime_data.anime_id.unique()
rated_anime_list = rating_data.anime_id.unique()
user_list = rating_data.user_id.unique()
                 
print('anime_count: {}, rated_anime_count: {}, user_count: {}'.format(len(anime_list), len(rated_anime_list), len(user_list)))
rating_data.head()
anime_data.head()
# drop watched but not rated scores
# TODO: replace with avg score 
rating_data.drop(rating_data[rating_data.rating == -1].index, inplace=True)
rating_data.reset_index(drop=True, inplace=True)

# Histogram of ratings
plt.hist(rating_data.rating)
plt.show()
print('Min ID: {}, Max ID: {}, count: {}'.format(anime_data.anime_id.min(), anime_data.anime_id.max(), anime_data.anime_id.count()))
# We won't be able to index items through their ID since we would get memory indexing errors. 
# To amend we can create a column that shows the spot in our list that particular movie is in:

anime_data['anime_index'] = anime_data.index
anime_data.head()
data_combined = pd.merge(rating_data[['user_id', 'anime_id', 'rating']], anime_data[['anime_id', 'anime_index']], on='anime_id')
data_combined.head()
data_grouped = data_combined.groupby('user_id')
data_grouped.first().head()
"""
Formatting the data into input for the RBM. 
Store the normalized users ratings into a list of lists called trX.
"""

max_rating = rating_data.rating.max()

# Amount of users used for training
amountOfUsedUsers = 1000

# Creating the training list
trX = []

# For each user in the group
for userID, curUser in data_grouped:

    # Create a temp that stores every movie's rating
    temp = [0]*len(anime_list)

    # For each anime in curUser's list
    for num, anime in curUser.iterrows():

        # Divide the rating by max and store it
        temp[anime['anime_index']] = anime.rating/max_rating

    # Add the list of ratings into the training list
    trX.append(temp)

    # Check to see if we finished adding in the amount of users for training
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1
# Check that user_id 0 have rated anime with index 1709
trX[0][1709]
# Setting the models Parameters
hiddenUnits = 50
visibleUnits = len(anime_list)
vb = tf.placeholder(tf.float32, [visibleUnits])  # Number of unique animes
hb = tf.placeholder(tf.float32, [hiddenUnits])  # Number of features were going to learn
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix
# Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling
# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
""" Set RBM Training Parameters """

# Learning rate
alpha = 1.0

# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set the error function, here we use Mean Absolute Error Function
err = v0 - v1
err_sum = tf.reduce_mean(err*err)
""" Initialize our Variables with Zeroes using Numpy Library """

# Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)

# Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)

# Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)

# Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Train RBM with 15 Epochs, with Each Epoch using 10 batches with size 100, After training print out the error by epoch
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()
"""
Recommendation System :-
- We can now predict anime that an arbitrarily selected user might like. 
- This can be accomplished by feeding in the user's watched preferences into the RBM and then reconstructing the 
  input. 
- The values that the RBM gives us will attempt to estimate the user's preferences for items that he hasn't watched 
  based on the preferences of the users that the RBM was trained on.
"""

# Select the example User
inputUser = [trX[50]]
# Feeding in the User and Reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
recommendation = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
# List the 20 most recommended items for our mock user by sorting it by their scores given by our model.
recos_for_user = anime_data
recos_for_user["Recommendation Score"] = recommendation[0]
recos_for_user.sort_values(["Recommendation Score"], ascending=False).head(20)
""" Recommend User what movies he has not watched yet """

# Find the mock user's UserID from the data
data_combined.iloc[50]  # Result you get is UserID 191
# Find all items the mock user has watched before
rated_anime_for_user = data_combined[data_combined['user_id'] == 191]
rated_anime_for_user.head()
""" Merge all items that our mock users has watched with predicted scores based on his historical data: """

# Merging scored items with rated items by ID
merge_watched_recommended = recos_for_user.merge(rated_anime_for_user, on='anime_id', how='outer')
merge_watched_recommended.head()
# Dropping unnecessary columns
merge_watched_recommended = merge_watched_recommended.drop('anime_index_y', axis=1).drop('user_id', axis=1)
# Sort and take a look at first 20 rows
merge_watched_recommended.sort_values(['Recommendation Score'], ascending=False).head(20)

# """ There are some item the user has not watched and has high score based on our model. So, we can recommend them. """