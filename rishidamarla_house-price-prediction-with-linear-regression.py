# Importing all necessary libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Loading the data.

from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
X
y
X.shape
y.shape
# Implementing the Linear Regression model.

from sklearn import linear_model

reg = linear_model.LinearRegression()
# Splitting the data into training and testing sets.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
reg.fit(X_train, y_train)
# Printing out the predicted values.

y_pred = reg.predict(X_test)

print(y_pred)
# Printing out the actual values.

print(y_test)
# Printing out how well the model performed.

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, y_pred))