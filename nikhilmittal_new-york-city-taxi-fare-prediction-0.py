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
# Reading first 10M training data
train_df = pd.read_csv('../input/train.csv', nrows=10000000)
train_df.dtypes
# Adding Manhattan Distance as features
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)

print(train_df.isnull().sum())
# Given that nulls are lesser in number, let's remove these from the training dataset
print('Old size: %d' %len(train_df))
train_df = train_df.dropna(how='any', axis='rows')
print('New size: %d' %len(train_df))
# Let's plot a subset of travel vector feature we added to see it's distribution
plot = train_df.iloc[:4000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' %len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' %len(train_df))
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))
                           
train_X = get_input_matrix(train_df)
train_y = np.array(train_df['fare_amount'])
                           
print(train_X.shape)
print(train_y.shape)
# Now let's use lstsq function to find the optimal weights
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
print(w)
w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)
print(w_OLS)
test_df = pd.read_csv('../input/test.csv')
test_df.dtypes
add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)

test_y_predictions = np.matmul(test_X, w).round(decimals=2)

submission = pd.DataFrame({'key':test_df.key, 'fare_amount':test_y_predictions}, 
                          columns=['key', 'fare_amount'])

submission.to_csv('submission.csv', index=False)

print(os.listdir('.'))