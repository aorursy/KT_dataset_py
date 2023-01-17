import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error

import seaborn as sns
data = pd.read_csv('../input/50-startups/50_Startups.csv')

data = data.drop(['State'], axis=1)
x = data.drop(['Profit'], axis=1)

y = data['Profit']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The MSE score is {mean_squared_error(y_test, y_pred)}')

print(model.coef_)
plt.scatter(y_train, model.predict(X_train))
plt.scatter(y_test, model.predict(X_test))
from sklearn.linear_model import Ridge

model = Ridge()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The MSE score is {mean_squared_error(y_test, y_pred)}')

print(model.coef_)
plt.scatter(y_train, model.predict(X_train))
plt.scatter(y_test, model.predict(X_test))
from sklearn.linear_model import Lasso

model = Lasso()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The MSE score is {mean_squared_error(y_test, y_pred)}')

print(model.coef_)
plt.scatter(y_train, model.predict(X_train))
plt.scatter(y_test, model.predict(X_test))
from sklearn.linear_model import ElasticNet

model = ElasticNet()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The MSE score is {mean_squared_error(y_test, y_pred)}')

print(model.coef_)
plt.scatter(y_train, model.predict(X_train))
plt.scatter(y_test, model.predict(X_test))
from sklearn.linear_model import BayesianRidge

model = BayesianRidge()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'The R2 score is: {r2_score(y_test, y_pred)}')

print(f'The MSE score is {mean_squared_error(y_test, y_pred)}')

print(model.coef_)
plt.scatter(y_train, model.predict(X_train))
plt.scatter(y_test, model.predict(X_test))