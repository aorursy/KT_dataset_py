import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

from sklearn import linear_model
df=pd.read_csv("../input/home-prices-in-usa/Home prices in USA.csv")

df
plt.xlabel('area(sq ft)')

plt.ylabel('price(US $)')

plt.scatter(df.area,df.price)
X = df.drop('price', axis = 1)

Y = df['price']
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
reg = linear_model.LinearRegression()

reg.fit(X_train, Y_train)

Y_pred=reg.predict(X_test)

plt.scatter(Y_test, Y_pred)
