import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression
data=pd.read_csv("../input/machinelearning/studentscores.csv")
data.head()
X=data.iloc[:, : 1].values

Y=data.iloc[:, 1].values
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.25, random_state=0)
regressor= LinearRegression()

regressor= regressor.fit(X_train, Y_train)
# predict result



y_pred=regressor.predict(X_test)
# visualization

#training

import matplotlib.pyplot as plt

plt.scatter(X_train, Y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')
#testing

plt.scatter(X_test, Y_test, color='red')

plt.plot(X_test, regressor.predict(X_test), color='blue')