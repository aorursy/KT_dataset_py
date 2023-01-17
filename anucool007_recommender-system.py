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
headings = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('../input/userData.csv', names=headings)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print ('No. of Users = ' + str(n_users) + ' | no. of Movies = ' + str(n_items))
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)
#Memory-Based Collaborative Filtering
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples(): train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples(): test_data_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')
print (item_prediction[[1,2]])
print (user_prediction[[1,2]])
#Evaluation
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
print ('User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
#Model-based Collaborative Filtering
import scipy.sparse as sp
from scipy.sparse.linalg import svds
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print ('The sparsity level the dataset is ' +  str(sparsity*100) + '%')

u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print(X_pred[[1,2]])
print ('User based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))
