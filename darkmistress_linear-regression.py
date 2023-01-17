#Import libraries

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
#load dataset

bos = pd.read_csv('../input/housing/boston.csv')
bos.head(5)
#Create features and target arrays

X = bos.drop('MEDV', axis=1).values

y = bos['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.35)
#Fit a regression model

reg = LinearRegression()

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(y_pred)
reg.score(X_test, y_test)