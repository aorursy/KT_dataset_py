# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import warnings

import itertools

from pylab import rcParams

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
device_lib.list_local_devices()
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print('Shape of the training data with all features is:', train.shape)



print(train.dtypes)

train = train.select_dtypes(exclude=['object'])

print('Shape of the newly formed training data with all features is:', train.shape)
train.drop(['Id'],axis = 1, inplace = True)

train.fillna(0,inplace=True)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test = test.select_dtypes(exclude=['object'])

ID = test.Id

test.fillna(0,inplace=True)

test.drop('Id',axis = 1, inplace = True)
print("List of features remaining in our dataset:",list(train.columns))
train.head()
from sklearn.ensemble import IsolationForest
#implementing the Isolation Forest algorithm

iso_for = IsolationForest(max_samples = 100, random_state = 50)

iso_for.fit(train)



y = iso_for.predict(train)

#creating a dataframe with a single Top Column

y = pd.DataFrame(y, columns = ['Top'])



#selecting all the values that are non-outliers and setting them to training dataset

train = train.iloc[y[y['Top'] == 1].index.values]

#resetting the index of our training dataset

train.reset_index(drop = True, inplace = True)



#finding out our outliers and non-outliers

print("Number of Outliers:", y[y['Top'] == -1].shape[0])

print("Number of rows without outliers:", train.shape[0])
col_train = list(train.columns)

col_train_bis = list(train.columns)



col_train_bis.remove('SalePrice')



mat_train = np.matrix(train)

mat_test  = np.matrix(test)

mat_new = np.matrix(train.drop('SalePrice',axis = 1))

mat_y = np.array(train.SalePrice).reshape((1314,1))



prepro_y = MinMaxScaler()

prepro_y.fit(mat_y)



prepro = MinMaxScaler()

prepro.fit(mat_train)



prepro_test = MinMaxScaler()

prepro_test.fit(mat_new)



train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)

test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)



train.head()
# List of features

COLUMNS = col_train

FEATURES = col_train_bis

LABEL = "SalePrice"

# Columns for simplified model for Tensorflow use



feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
training_set = train[COLUMNS]        #defining our training data

prediction_set = train.SalePrice     #defining our predicution feature
#performing our train test split



X_train, X_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.33, random_state=50)

y_train = pd.DataFrame(y_train, columns = [LABEL])

training_set = pd.DataFrame(X_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)

training_set.head()



# Training for submission

training_sub = training_set[col_train]
y_test = pd.DataFrame(y_test, columns = [LABEL])

testing_set = pd.DataFrame(X_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)

testing_set.head()
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 

                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12],

                                          optimizer = tf.train.GradientDescentOptimizer( learning_rate= 0.1 ))
# Reset the index of training

training_set.reset_index(drop = True, inplace =True)
def input_function(data_set, pred = False):

    

    if pred == False:

        

        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

        labels = tf.constant(data_set[LABEL].values)

        

        return feature_cols, labels



    if pred == True:

        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

        

        return feature_cols
# Deep Neural Network Regressor with the training set which contain the data split by train test split

regressor.fit(input_fn=lambda: input_function(training_set), steps=2000)
# Evaluation on the test set created by train_test_split

evaluate = regressor.evaluate(input_fn=lambda: input_function(testing_set), steps=1)
# Display the score on the testing set

loss_score = evaluate["loss"]

print("Final Loss on the testing set: {0:f}".format(loss_score))
y_final = regressor.predict(input_fn=lambda: input_function(testing_set))

predictions = list(itertools.islice(y, testing_set.shape[0]))
y = regressor.predict(input_fn=lambda: input_function(testing_set))

predictions = list(itertools.islice(y, testing_set.shape[0]))
y_predict = regressor.predict(input_fn=lambda: input_function(test, pred = True))



def to_submit(pred_y,name_out):

    y_predict = list(itertools.islice(pred_y, test.shape[0]))

    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict),1)), columns = ['SalePrice'])

    y_predict = y_predict.join(ID)

    y_predict.to_csv(name_out + '.csv',index=False)

    

to_submit(y_predict, "submission_final_PK")