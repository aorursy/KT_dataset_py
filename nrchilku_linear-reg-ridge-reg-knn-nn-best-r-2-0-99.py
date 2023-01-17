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
data =  pd.read_csv('../input/roboBohr.csv')
train_x = data.drop(['Unnamed: 0', 'pubchem_id', 'Eat'], axis = 1)
train_y = data['Eat']
data.head(3)
# Changes the mean and standard deviation of features to 0 and 1 respectively.
def standardize(df):
    for column in range(0, df.shape[1]):
        df[str(column)] = (df[str(column)] - np.mean(df[str(column)]))/np.std(df[str(column)]) 
        
standardize(train_x)        
from sklearn.model_selection import train_test_split
train_x_1, val_x, train_y_1, val_y = train_test_split(train_x, train_y, test_size=0.3)
print(train_x_1.shape, val_x.shape)
# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(train_x_1, train_y_1)
print(lr.score(train_x_1, train_y_1), lr.score(val_x, val_y))
# Ridge Regression i.e., linear regression with L2 regularization.
from sklearn.linear_model import Ridge
ridge_models = {}
for i in [0.001, 0.1, 0.5, 1, 5, 10, 20, 30, 50, 60, 70, 100, 110, 120, 130, 140, 150, 200]:
    ridge_models[str(i)] = Ridge(alpha=i)
    ridge_models[str(i)].fit(train_x_1, train_y_1)
    print(i , ridge_models[str(i)].score(train_x_1, train_y_1), ridge_models[str(i)].score(val_x, val_y))
# Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
knn_models = {}
for i in range(1, 10):
    knn_models[str(i)] = KNeighborsRegressor(n_neighbors=i);
    knn_models[str(i)].fit(train_x_1, train_y_1);
    score = knn_models[str(i)].score(val_x, val_y)
    print("Validation score for n_neighbors = " + str(i) + " is " + str(score))
# Neural Networks
from sklearn.neural_network import MLPRegressor
nn_models = {}
for i in range(4,9):
    for j in range(1,6):
        nn_models[str((i,j))] = MLPRegressor((i,j), activation='relu', learning_rate='adaptive')
        nn_models[str((i,j))].fit(train_x_1, train_y_1)
        print((i,j), nn_models[str((i,j))].score(train_x_1, train_y_1), nn_models[str((i,j))].score(val_x, val_y))
final_model = KNeighborsRegressor(n_neighbors=3)
final_model.fit(train_x, train_y)