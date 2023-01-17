import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,8
train = pd.read_csv('../input/into-the-future/train.csv')
test = pd.read_csv('../input/into-the-future/test.csv')
print("The shape of the training set is {}".format(train.shape))
print("The shape of the test set is {}".format(test.shape))
train.isnull().sum()
test.isnull().sum()
train.dtypes
train['time'] = pd.to_datetime(train['time'])
train.index = train.time
train = train.drop(['time'], axis = 1)
train = train.drop(['id'], axis = 1)
train.head()
X = train.iloc[:,0:1].values
y = train.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_score = r2_score(y_test, y_pred_lr)
error = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("R squared value for the Linear Regression model is {} and the root mean squared error in prediction is {}".format(r2_score, error))
from sklearn.model_selection import cross_val_score
score = cross_val_score(lr, X_train, y_train, cv = 10)
mean = score.mean()
print("The mean cross validation score for the model for 10 models is {}".format(mean))
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4)
X_poly = pr.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 0)
lr2 = LinearRegression()
lr2.fit(X_train, y_train)
y_pred_poly = lr2.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_score = r2_score(y_test, y_pred_poly)
error = np.sqrt(mean_squared_error(y_test, y_pred_poly))
print("The r squared value for the Polynomial Regression model is {} and the root mean squared error in prediction is {}".format(r2_score, error))
from sklearn.model_selection import cross_val_score
score = cross_val_score(lr2, X_test, y_test, cv = 10)
mean = score.mean()
print("The mean cross validation score for the model for 10 models is {}".format(mean))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_score = r2_score(y_test, y_pred_svr)
error = np.sqrt(mean_squared_error(y_test, y_pred_svr))
print("The r squared value for the Support Vector Regression model is {} and the root mean squared error in prediction is {}".format(r2_score, error))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_score = r2_score(y_test, y_pred_rf)
error = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("The r squared value for the Random Forest Regressor model is {} and the root mean squared error in prediction is {}".format(r2_score, error))
score = cross_val_score(rf, X_train, y_train, cv = 10)
mean = score.mean()
print("The mean cross validation score for the model for 10 models is {}".format(mean))
n_estimators = [200,600,800,1000]
max_features = ['auto', 'sqrt']
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
parameters = {'n_estimators': n_estimators,
               'max_features': max_features}
from sklearn.model_selection import GridSearchCV
randomforest = GridSearchCV(estimator = rf, param_grid = parameters, cv = 10, n_jobs = -1)
randomforest.fit(X_train, y_train)
randomforest.best_params_
y_pred_rf = randomforest.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_score = r2_score(y_test, y_pred_rf)
error = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("The r squared value for the Random Forest Regressor model is {} and the root mean squared error in prediction is {}".format(r2_score, error))
score = cross_val_score(rf, X_train, y_train, cv = 10)
mean = score.mean()
print("The mean cross validation score for the model for 10 models is {}".format(mean))
from xgboost import XGBRegressor
XGB = XGBRegressor()
XGB.fit(X_train, y_train)
y_pred_xgb = XGB.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_score = r2_score(y_test, y_pred_xgb)
error = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("The r squared value for the XGBoost Regressor model is {} and the root mean squared error in prediction is {}".format(r2_score, error))
test = pd.read_csv('../input/into-the-future/test.csv')
test['time'] = pd.to_datetime(test['time'])
test.index = test.time
test = test.drop(['time'], axis = 1)
test.head()
X = test.iloc[:,1:2].values
test['predictions'] = randomforest.predict(X)
ids = test['id']
feature_2 = test['predictions']
submission = pd.DataFrame({"id": ids, "feature_2": feature_2})
submission.to_csv('Prediction2.csv')