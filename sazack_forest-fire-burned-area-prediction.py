# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/forestfires.csv')
dataset.head()
dataset.describe(include='all')
dataset.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataset.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace =True)
dataset.head()
corr = dataset.corr(method='pearson')
print("Correlation of the Dataset:",corr)
f,ax = plt.subplots(figsize=(18, 18))
print("Plotting correlation:")
sns.heatmap(corr,annot= True, linewidths=.5)

data = dataset.values

X = data[:,0:12]
Y = data[:,12]
extraTreesRegressor = ExtraTreesRegressor()
rfe = RFE(extraTreesRegressor,5)
fit = rfe.fit(X,Y)

print("The number of features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Rankings:", fit.ranking_)
dataset.plot(kind='density', subplots=True, layout=(4,4))
print("Linear Regression")
Lreg = LinearRegression()
Lreg.fit(X,Y)
prediction = Lreg.predict(X)
score = explained_variance_score(Y, prediction)
mae = mean_absolute_error(prediction, Y)

print("Score:", score)
print("Mean Absolute Error:", mae)
print("Lasso Regression")
lasso = Lasso()
lasso.fit(X,Y)
prediction_lasso = lasso.predict(X)
score = explained_variance_score(Y, prediction_lasso)
mae = mean_absolute_error(prediction_lasso, Y)

print("Score:", score)
print("Mean Absolute Error:", mae)
print("Ridge Regression")
ridge = Ridge()
ridge.fit(X,Y)
prediction_ridge = ridge.predict(X)
score = explained_variance_score(Y, prediction_ridge)
mae = mean_absolute_error(prediction_ridge, Y)

print("Score:", score)
print("Mean Absolute Error:", mae)
print('K-Neighbors Regressor')
knreg = KNeighborsRegressor()
knreg.fit(X,Y)
prediction_knreg = knreg.predict(X)
score = explained_variance_score(Y, prediction_knreg)
mae = mean_absolute_error(prediction_knreg, Y)

print("Score:", score)
print("Mean Absolute Error:", mae)

print('Random Forest Regressor')
rfreg = RandomForestRegressor()
rfreg.fit(X,Y)
prediction_rfreg = rfreg.predict(X)
score = explained_variance_score(Y, prediction_rfreg)
mae = mean_absolute_error(prediction_rfreg, Y)

print("Score:", score)
print("Mean Absolute Error:", mae)
print('Support Vector Regressor')
svr = SVR()
svr.fit(X,Y)
prediction_svr = svr.predict(X)
score = explained_variance_score(Y, prediction_svr)
mae = mean_absolute_error(prediction_svr, Y)

print("Score:", score)
print("Mean Absolute Error:", mae)