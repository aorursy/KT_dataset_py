from sklearn.datasets import load_boston
import numpy as np

import pandas as pd

import matplotlib as plt 
df = load_boston()

df
dataset = pd.DataFrame(df.data)  # Independent features

dataset.head()
dataset.columns = df.feature_names

dataset.head()
df.target.shape
dataset['Price'] = df.target

dataset.head()
X = dataset.iloc[:,:-1] # independent features

y = dataset.iloc[:,-1] # dependent features
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression



linear_regressor = LinearRegression()

mse = cross_val_score(linear_regressor, X, y, scoring='neg_mean_squared_error', cv=5)

mean_mse = np.mean(mse)

print(mean_mse)
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



ridge = Ridge()

parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(X, y)



print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



lasso = Lasso()

parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(X, y)



print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
prediction_lasso = lasso_regressor.predict(X_test)

prediction_ridge = ridge_regressor.predict(X_test)
import seaborn as sns



sns.distplot(y_test-prediction_lasso)
sns.distplot(y_test-prediction_ridge)