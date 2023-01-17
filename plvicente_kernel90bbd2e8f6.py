# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import numpy.random as nr

import math

from collections import Counter, OrderedDict

from datetime import date

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data_train = pd.read_csv('../input/train_MV.csv', nrows = 55_000_000)

data_train.dtypes

data_train.head()



data_test = pd.read_csv('../input/test_MV.csv', nrows = 10_000)

data_test.dtypes

data_test.head()



data_submission = pd.read_csv('../input/submission_MV.csv')

data_submission.dtypes

data_submission.head()



# Any results you write to the current directory are saved as output.

def add_traveller_feat(df):

     df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

     df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

     add_traveller_feat(data_train)

     

print(data_train.isnull().sum())



print('Old size: %d' % len(data_train))

data_train = data_train.dropna(how = 'any', axis = 'rows')

print('New size: %d' % len(data_train))



#plot_test = data_train.iloc[:2000].plot.scatter('data_train.pickup_longitude', 'data_train.pikckup_latitude')



print('Old size: %d' % len(data_train))

data_train = data_train[(data_train.pickup_longitude < 5.0) & (data_train.dropoff_latitude < 5.0)]

print('New size: %d' % len(data_train))



# Construct and return an Nx3 input matrix for our linear model

# using the travel vector, plus a 1.0 for a constant bias term.

def get_input_matrix(df):

    return np.column_stack((df.pickup_longitude, df.dropoff_latitude, np.ones(len(df))))



train_X = get_input_matrix(data_train)

train_y = np.array(data_train['fare_amount'])



print(train_X.shape)

print(train_y.shape)



# Reuse the above helper functions to add our features and generate the input matrix.

# The lstsq function returns several things, and we only care about the actual weight vector w.

(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)

print(w)

w_SLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)

print(w_SLS)



#add_traveller_feat(data_test)

test_X = get_input_matrix(data_test)

# Predict fare_amount on the test set using our model (w) trained on the training set.

test_y_predictions = np.matmul(test_X, w).round(decimals = 2)



# Write the predictions to a CSV file which we can submit to the competition.

submission = pd.DataFrame(

    {'key': data_test.key, 'fare_amount': test_y_predictions},

    columns = ['key', 'fare_amount'])

submission.to_csv('submissionMV.csv', index = False)



print(os.listdir('.'))



print ("testing...")