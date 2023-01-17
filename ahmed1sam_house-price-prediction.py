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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/kc_house_data.csv')
dataset.head()
X = dataset.iloc[:, 3:].values
print (X[0,:])
y = dataset.iloc[:, 2].values
print (y[0])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print (y_pred[0:10])
X_input=[[3,1.00,1180,5650,1.0,0,0,3,7,1180,0,1955,0,98178,47.5112,-122.257,1340,5650]]
print (regressor.predict(X_input))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)

print ("accurancy = ", accuracies.mean())
print ("Std=",accuracies.std())
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_MLR = LinearRegression()
regressor_MLR.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor_MLR, X = X_train, y = y_train, cv = 10)

print ("accurancy = ", accuracies.mean())
print ("Std=",accuracies.std())

from sklearn.svm import SVR
regressor_SVM = SVR(kernel = 'rbf')
regressor_SVM.fit(X, y)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor_SVM, X = X_train, y = y_train, cv = 10)

print ("accurancy = ", accuracies.mean())
print ("Std=",accuracies.std())

