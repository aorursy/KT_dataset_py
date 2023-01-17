import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from subprocess import check_output

dataset_train = pd.read_csv('../input/train.csv')
dataset_train.head()
grLivArea = dataset_train.loc[:, 'GrLivArea']
totalBsmtSF = dataset_train.loc[:, 'TotalBsmtSF']
X = totalBsmtSF + grLivArea

y = dataset_train.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
plt.scatter(X_train,y_train)
plt.show()
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.values.reshape(-1, 1)
regressor.fit(X_train, y_train)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
regressor.score(X_train, y_train)
X_test = X_test.values.reshape(-1, 1)
regressor.score(X_test, y_test)
X = dataset_train.iloc[:, 1:-1]
y = dataset_train.iloc[:, -1]
X = pd.get_dummies(X)
X = X.fillna(X.mean())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)