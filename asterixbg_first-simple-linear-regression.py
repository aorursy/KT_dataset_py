# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/housing.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 3].values



# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#Training Regression Model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

pred = regressor.fit(X_train, y_train)

y_pred = pred.predict(X_test)



plt.plot(y_pred)

plt.plot(y_test)

plt.show()