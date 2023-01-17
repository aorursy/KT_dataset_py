from sklearn.datasets import load_boston

from sklearn import preprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from prettytable import PrettyTable

from sklearn.linear_model import SGDRegressor

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from numpy import random

from sklearn.model_selection import train_test_split
boston_data=pd.DataFrame(load_boston().data,columns=load_boston().feature_names)

Y=load_boston().target

X=load_boston().data

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
# data overview

boston_data.head(3)
print(X.shape)

print(Y.shape)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
## Before standardizing data

x_train
# standardizing data

scaler = preprocessing.StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test=scaler.transform(x_test)
## After standardizing data

x_train

x_test
## Adding the PRICE Column in the data

train_data=pd.DataFrame(x_train)

train_data['price']=y_train

train_data.head(3)
x_test=np.array(x_test)

y_test=np.array(y_test)
type(x_test)
n_iter=100
# SkLearn SGD classifier

clf_ = SGDRegressor(max_iter=n_iter)

clf_.fit(x_train, y_train)

y_pred_sksgd=clf_.predict(x_test)

plt.scatter(y_test,y_pred_sksgd)

plt.grid()

plt.xlabel('Actual y')

plt.ylabel('Predicted y')

plt.title('Scatter plot from actual y and predicted y')

plt.show()
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_sksgd))
# SkLearn SGD classifier predicted weight matrix

sklearn_w=clf_.coef_

sklearn_w
type(sklearn_w)