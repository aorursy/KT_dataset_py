# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn import decomposition

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

#path of the file

data = "../input/WeatherData.csv"



WD = pd.read_csv(data)



# Any results you write to the current directory are saved as output.
# Creating the Prediction variable

y = WD.actual



# Reviewing the columns in data

print(WD.describe())

print(WD.shape)

WD = WD.drop('actual',1)





print(WD.shape)

# print(WD.dtypes.sample(10))

# one_hot_encoded_training_predictors = pd.get_dummies(WD)

# print(one_hot_encoded_training_predictors)

# print(WD)

# X = one_hot_encoded_training_predictors

# print(X.shape)

#choosing prediction features, X

# features = ['year','month','day','temp_2','temp_1', 'average']

X = WD.drop('week',1)

# print(X)



# data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = MinMaxScaler()

scaler.fit(X)

# MinMaxScaler(copy=True, feature_range=(0, 1))

X = scaler.transform(X)

# [[0.   0.  ]

#  [0.25 0.25]

#  [0.5  0.5 ]

#  [1.   1.  ]]



pca = decomposition.PCA(n_components=5)

pca.fit(X)

X = pca.transform(X)

print(X.shape)

print(y.shape)

#Spliting data into training and validation 

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)



print(train_X.shape)

print(val_X.shape)

print(train_y.shape)

print(val_y.shape)



# Defining the model

temp_model = RandomForestRegressor(random_state = 1)

#temp_model = DecisionTreeRegressor



# Fitting the model

temp_model.fit(train_X, train_y)
temp_pred = temp_model.predict(val_X)

val_mae = mean_absolute_error(temp_pred, val_y)

#print(temp_pred) 

#print(val_y)

print(val_mae)
#calculate errors

errors = abs (temp_pred-val_y)

#print (errors)



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / val_y)

#print(mape)



# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')