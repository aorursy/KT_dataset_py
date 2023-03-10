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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install tensorflow==1.14

import random

from collections import Counter

from sklearn.metrics import roc_curve, auc, average_precision_score
import tensorflow as tf

tf.__version__
path = '../input/steam-games-2017-cleaned/steam-200k-cleaned.csv'

#path = 'steam-200k.csv'

df = pd.read_csv(path, header = None,

                 names = ['UserID', 'Game', 'Action', 'Hours', 'Other'])

df.head()



df['Hours_Played'] = df['Hours'].astype('float32')



df.loc[(df['Action'] == 'purchase') & (df['Hours'] == 1.0), 'Hours_Played'] = 0
df.UserID = df.UserID.astype('int')

df = df.sort_values(['UserID', 'Game', 'Hours_Played'])



clean_df = df.drop_duplicates(['UserID', 'Game'], keep = 'last').drop(['Action', 'Hours', 'Other'], axis = 1)



# every transaction is represented by only one record now

clean_df.head()
n_users = len(clean_df.UserID.unique())

n_games = len(clean_df.Game.unique())



print('There are {0} users and {1} games in the data'.format(n_users, n_games))
sparsity = clean_df.shape[0] / float(n_users * n_games)

print('{:.2%} of the user-item matrix is filled'.format(sparsity))
user_counter = Counter()

for user in clean_df.UserID.tolist():

    user_counter[user] +=1



game_counter = Counter()

for game in clean_df.Game.tolist():

    game_counter[game] += 1



user2idx = {user: i for i, user in enumerate(clean_df.UserID.unique())}

idx2user = {i: user for user, i in user2idx.items()}



game2idx = {game: i for i, game in enumerate(clean_df.Game.unique())}

idx2game = {i: game for game, i in game2idx.items()}
user_idx = clean_df['UserID'].apply(lambda x: user2idx[x]).values

game_idx = clean_df['gameIdx'] = clean_df['Game'].apply(lambda x: game2idx[x]).values

hours = clean_df['Hours_Played'].values
zero_matrix = np.zeros(shape = (n_users, n_games)) # Create a zero matrix

user_game_pref = zero_matrix.copy()

user_game_pref[user_idx, game_idx] = 1 # Fill preference matrix



user_game_interactions = zero_matrix.copy()

# Confidence matrix

user_game_interactions[user_idx, game_idx] = hours + 1 
#??????



k = 5



# Count the number of purchases for each user

purchase_counts = np.apply_along_axis(np.bincount, 1, user_game_pref.astype(int))

buyers_idx = np.where(purchase_counts[:, 1] >= 2 * k)[0] #find the users who purchase 2 * k games

print('{0} users bought {1} or more games'.format(len(buyers_idx), 2 * k))
test_frac = 0.2 # Let's save 10% of the data for validation and 10% for testing.

test_users_idx = np.random.choice(buyers_idx,

                                  size = int(np.ceil(len(buyers_idx) * test_frac)),

                                  replace = False)
val_users_idx = test_users_idx[:int(len(test_users_idx) / 2)]

test_users_idx = test_users_idx[int(len(test_users_idx) / 2):]
def data_process(dat, train, test, user_idx, k):

    for user in user_idx:

        purchases = np.where(dat[user, :] == 1)[0]

        mask = np.random.choice(purchases, size = k, replace = False)

        

        train[user, mask] = 0

        test[user, mask] = dat[user, mask]

    return train, test
train_matrix = user_game_pref.copy()

test_matrix = zero_matrix.copy()

val_matrix = zero_matrix.copy()



# Mask the train matrix and create the validation and test matrices

train_matrix, val_matrix = data_process(user_game_pref, train_matrix, val_matrix, val_users_idx, k)

train_matrix, test_matrix = data_process(user_game_pref, train_matrix, test_matrix, test_users_idx, k)
test_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]
train_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]
tf.reset_default_graph() # Create a new graphs



pref = tf.placeholder(tf.float32, (n_users, n_games))  # Here's the preference matrix

interactions = tf.placeholder(tf.float32, (n_users, n_games)) # Here's the hours played matrix

users_idx = tf.placeholder(tf.int32, (None))
n_features = 30 # ???????????? ?????? ???



# The X matrix represents the user latent preferences with a shape of user x latent features

X = tf.Variable(tf.truncated_normal([n_users, n_features], mean = 0, stddev = 0.05))



# The Y matrix represents the game latent features with a shape of game x latent features

Y = tf.Variable(tf.truncated_normal([n_games, n_features], mean = 0, stddev = 0.05))



# Here's the initilization of the confidence parameter

conf_alpha = tf.Variable(tf.random_uniform([1], 0, 1))
#user bias

user_bias = tf.Variable(tf.truncated_normal([n_users, 1], stddev = 0.2))



# Concatenate the vector to the user matrix

X_plus_bias = tf.concat([X, 

                         #tf.convert_to_tensor(user_bias, dtype = tf.float32),

                         user_bias,

                         tf.ones((n_users, 1), dtype = tf.float32)], axis = 1)
# game bias

item_bias = tf.Variable(tf.truncated_normal([n_games, 1], stddev = 0.2))



# Cocatenate the vector to the game matrix

Y_plus_bias = tf.concat([Y, 

                         tf.ones((n_games, 1), dtype = tf.float32),

                         item_bias],

                         axis = 1)
pred_pref = tf.matmul(X_plus_bias, Y_plus_bias, transpose_b=True)



# Construct the confidence matrix with the hours played and alpha paramter

conf = 1 + conf_alpha * interactions
cost = tf.reduce_sum(tf.multiply(conf, tf.square(tf.subtract(pref, pred_pref))))

l2_sqr = tf.nn.l2_loss(X) + tf.nn.l2_loss(Y) + tf.nn.l2_loss(user_bias) + tf.nn.l2_loss(item_bias)

lambda_c = 0.01

loss = cost + lambda_c * l2_sqr
lr = 0.05

optimize = tf.train.AdagradOptimizer(learning_rate = lr).minimize(loss)
# This is a function that helps to calculate the top k precision 

def top_k_precision(pred, mat, k, user_idx):

    precisions = []

    

    for user in user_idx:

        rec = np.argsort(-pred[user, :]) # Found the top recommendation from the predictions

        

        top_k = rec[:k]

        labels = mat[user, :].nonzero()[0]

        

        precision = len(set(top_k) & set(labels)) / float(k) # Calculate the precisions from actual labels

        precisions.append(precision)

    return np.mean(precisions) 
iterations = 100

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    

    for i in range(iterations):

        sess.run(optimize, feed_dict = {pref: train_matrix,

                                        interactions: user_game_interactions})

        

        if i % 10 == 0:

            mod_loss = sess.run(loss, feed_dict = {pref: train_matrix,

                                                   interactions: user_game_interactions})            

            mod_pred = pred_pref.eval()

            train_precision = top_k_precision(mod_pred, train_matrix, k, val_users_idx)

            val_precision = top_k_precision(mod_pred, val_matrix, k, val_users_idx)

            print('Iterations {0}...'.format(i),

                  'Training Loss {:.2f}...'.format(mod_loss),

                  'Train Precision {:.3f}...'.format(train_precision),

                  'Val Precision {:.3f}'.format(val_precision)

                )



    rec = pred_pref.eval()

    test_precision = top_k_precision(rec, test_matrix, k, test_users_idx)

    print('\n')

    print('Test Precision{:.3f}'.format(test_precision))
n_examples = 10

users = np.random.choice(test_users_idx, size = n_examples, replace = False)

rec_games = np.argsort(-rec)
for user in users:

    print('User #{0} recommendations ...'.format(idx2user[user]))

    purchase_history = np.where(train_matrix[user, :] != 0)[0]

    recommendations = rec_games[user, :]



    

    new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]

    

    print('Recommendations')

    print(', '.join([idx2game[game] for game in new_recommendations]))

    print('\n')

    print('Actual purchases')

    print(', '.join([idx2game[game] for game in np.where(test_matrix[user, :] != 0)[0]]))

    print('\n')

    print('Precision of {0}'.format(len(set(new_recommendations) & set(np.where(test_matrix[user, :] != 0)[0])) / float(k)))

    print('--------------------------------------')

    print('\n')