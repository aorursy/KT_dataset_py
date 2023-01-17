# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd



# pass in column names for each CSV and read them using pandas. 

# Column names available in the readme file



#Reading users file:

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('../input/u.user', sep='|', names=u_cols,

 encoding='latin-1')



#Reading ratings file:

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings = pd.read_csv('../input/u.data', sep='\t', names=r_cols,

 encoding='latin-1')



#Reading items file:

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',

 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',

 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('../input/u.item', sep='|', names=i_cols,

 encoding='latin-1')
print (users.shape)

users.head()
print (ratings.shape)

ratings.head()
print (items.shape)

items.head()
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('../input/ua.base', sep='\t', names=r_cols,

 encoding='latin-1')



ratings_test = pd.read_csv('../input/ua.test', sep='\t', names=r_cols,

 encoding='latin-1')



ratings_base.shape, ratings_test.shape
import graphlab

train_data = graphlab.SFrame(ratings_base)

test_data = graphlab.SFrame(ratings_test)
from surprise import Reader, Dataset

# Define the format

reader = Reader(line_format='user item rating timestamp', sep='\t')

# Load the data from the file using the reader format

data = Dataset.load_from_file('../input/u.data', reader=reader)
# Split data into 5 folds

data.split(n_folds=5)
from surprise import SVD, evaluate

#From the famous available algorithms I mention SVD, NMF, KNN

algo = SVD()

#Root mean squared error (RMSE) and Mean absolute error (MAE).

evaluate(algo, data, measures=['RMSE', 'MAE'])
# Retrieve the trainset.

trainset = data.build_full_trainset()

algo.train(trainset)
userid = str(196)

itemid = str(302)

actual_rating = 4

print (algo.predict(userid, 302, 4))
header = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('../input/u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]

n_items = df.item_id.unique().shape[0]

print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
from sklearn import cross_validation as cv

train_data, test_data = cv.train_test_split(df, test_size=0.25)
#Create two user-item matrices, one for training and another for testing

train_data_matrix = np.zeros((n_users, n_items))

for line in train_data.itertuples():

    train_data_matrix[line[1]-1, line[2]-1] = line[3]



test_data_matrix = np.zeros((n_users, n_items))

for line in test_data.itertuples():

    test_data_matrix[line[1]-1, line[2]-1] = line[3]
from sklearn.metrics.pairwise import pairwise_distances

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
def predict(ratings, similarity, type='user'):

    if type == 'user':

        mean_user_rating = ratings.mean(axis=1)

        #You use np.newaxis so that mean_user_rating has same format as ratings

        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])

        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

    elif type == 'item':

        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    return pred
item_prediction = predict(train_data_matrix, item_similarity, type='item')

user_prediction = predict(train_data_matrix, user_similarity, type='user')
from sklearn.metrics import mean_squared_error

from math import sqrt

def rmse(prediction, ground_truth):

    prediction = prediction[ground_truth.nonzero()].flatten()

    ground_truth = ground_truth[ground_truth.nonzero()].flatten()

    return sqrt(mean_squared_error(prediction, ground_truth))
print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))

print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
sparsity=round(1.0-len(df)/float(n_users*n_items),3)

print ('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')
import scipy.sparse as sp

from scipy.sparse.linalg import svds



#get SVD components from train matrix. Choose k.

u, s, vt = svds(train_data_matrix, k = 20)

s_diag_matrix=np.diag(s)

X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

print ('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))