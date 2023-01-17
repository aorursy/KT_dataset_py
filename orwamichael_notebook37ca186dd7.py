import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')

df.head()
X = df['Temperature']

y = df['Revenue']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1,1))

y_pred = regressor.predict(X_test.values.reshape(-1,1))

data = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})

data
data.head()