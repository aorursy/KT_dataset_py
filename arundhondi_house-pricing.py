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
#importing additional lib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#data objects;
train_file_path='../input/train.csv'
test_file_path='../input/test.csv'
train_data=pd.read_csv(train_file_path)
test_data=pd.read_csv(test_file_path)


y=train_data['SalePrice']
X=train_data.drop(labels='SalePrice',axis=1)
print(test_data.info())

#feature function
def feature(X):
    colums=['LotArea','OverallCond','Id']
    X=X[colums]
    return X
X=feature(X)
test_data=feature(test_data)
print(format(X.shape))
#Splitting the data into train & and test set

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

# print(train_y.describe())
#defining the model
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(train_X,train_y)
prediction_train=model.predict(train_X)
predicted_prices = model.predict(test_data)
error_train=mean_absolute_error(train_y,prediction_train)
error_test=mean_absolute_error(val_y,prediction_test)
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('hp_submission.csv', index=False)
