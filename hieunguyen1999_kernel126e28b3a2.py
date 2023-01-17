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
from sklearn.metrics.pairwise import pairwise_distances

import math
# basic setup read data from movielens



rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']



ratings_base = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')

ratings_test = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/ua.test', sep='\t', names=rs_cols, encoding='latin-1')
# ratings_base.describe()



# Set numbers of user base and movie base

n_users_base = ratings_base['user_id'].unique().max()

n_items_base = ratings_base['movie_id'].unique().max()



n_users_base,n_items_base
# ratings_test.describe()



# Set numbers of user test and movie test

n_users_test = ratings_test['user_id'].unique().max()

n_items_test = ratings_test['movie_id'].unique().max()



n_users_test,n_items_test
# Create user - item matrix (row: user, column: item) start at 0



train_matrix = np.zeros((n_users_base, n_items_base))

for line in ratings_base.itertuples():

    train_matrix[line[1]-1,line[2]-1] = line[3]



test_matrix = np.zeros((n_users_test, n_items_test))

for line in ratings_test.itertuples():

    test_matrix[line[1]-1,line[2]-1] = line[3]

    

train_matrix
import scipy.sparse as sp

from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
# Get u,s,vt by using SVD with k = 20 

u, s, vt = svds(train_matrix, k = 20)



u.shape, s.shape, vt.shape
# Construction diagonal matrix

s_diag_matrix = np.diag(s)

s_diag_matrix
# Get the predictions by finding the dot product of the three matrices

# np.dot(a,b) returns the dot product of a and b.

predictions_svd = np.dot(np.dot(u,s_diag_matrix),vt)

# predictions_svd
# Get predicted ratings from predictions svd matrix

# trả về ma trận có rating khác 0, dựa trên ma trận test_matrix

# Ground truth target values. 

predicted_ratings_svd = predictions_svd[test_matrix.nonzero()]



# predicted_ratings_svd 



# Estimated target values.

test_truth = test_matrix[test_matrix.nonzero()]

# test_truth
# Calculate RMSE

RMSE = mean_squared_error(predicted_ratings_svd,test_truth,squared=False)

# Calculate MAE

MAE = mean_absolute_error(predicted_ratings_svd,test_truth)



RMSE, MAE
user_id = 1

user_ratings = predictions_svd[user_id-1,:]



# get all movies not ratings of user 1

train_unkown_indices = np.where(train_matrix[user_id-1,:] == 0)[0]



# get recommendations for user 1

user_recommendations = user_ratings[train_unkown_indices]



# train_unkown_indices

# user_recommendations.shape

user_ratings, train_unkown_indices, user_recommendations

# get numbers of movie ratings of user 1

# user_id : 1 , row : 0 

row_of_user = user_id -1;

train_matrix[train_matrix[[row_of_user]].nonzero()].shape
numbers_movie_rcm = 5 ;



print('\nRecommendations movies for user {} : \n'.format(user_id))



# Get recommendations of 5 movies with the highest scores



for movie_id in user_recommendations.argsort()[-numbers_movie_rcm:]:

    print(movie_id +1)