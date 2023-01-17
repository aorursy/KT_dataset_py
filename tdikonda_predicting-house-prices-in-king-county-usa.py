# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read csv file
kc = pd.read_csv('../input/kc_house_data.csv')
# set maximum number of columns to display
pd.set_option('display.max_columns',25)
# check data contents
kc.head()
# check data types
kc.dtypes
# check if there are any null values
kc.isnull().sum()
# Dropping 'id' and 'date' columns as they are not relevant to us
kc = kc.drop(columns=['id', 'date'])
# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Set figure size for seaborn
sns.set(rc={'figure.figsize':(20,10)})
# heatmap for correlation of columns
sns.heatmap(kc.corr(), annot=True, cmap="YlGnBu")
# dropping 'condition', 'yr_renovated' and 'zipcode' as they do not seem to have high correlation
kc = kc.drop(columns=['condition', 'yr_renovated', 'zipcode'])
# Check distribution of 'price'
sns.distplot(kc.price)
sns.scatterplot(x='price', y='sqft_living',data=kc)
# Do train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(kc.drop(columns='price'), kc.price, test_size=0.3)
# Import Regression algorithms
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, ElasticNet, RANSACRegressor, SGDRegressor, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
regressors = [DecisionTreeRegressor( ), LinearRegression( ), Lasso(max_iter=2000), Ridge( ), RidgeCV( ),ElasticNet(max_iter=2000), RANSACRegressor( ), SGDRegressor(max_iter=1000, tol=1e-3), HuberRegressor( ), BayesianRidge( ), AdaBoostRegressor(n_estimators=250, learning_rate=0.1), GradientBoostingRegressor( ), RandomForestRegressor(n_estimators=100), BaggingRegressor( ), ExtraTreesRegressor(n_estimators=100), SVR(gamma='scale'), KNeighborsRegressor( ), XGBRegressor( ), MLPRegressor(max_iter=500)]

for rgrs in regressors:
    rgrs.fit(X_train, y_train)
    name = rgrs.__class__.__name__
    
    print('='*30)
    print(name)
    print('****Results****')
    preds = rgrs.predict(X_test)
    print(f'Score : {rgrs.score(X_test,y_test)}')
    print(f'RMSE : {mean_squared_error(y_test, preds)**0.5}')
# Import PermutationImportance which provides insight into feature importance
import eli5
from eli5.sklearn import PermutationImportance
# Feature importance for ExtraTreesRegressor
etr_model = ExtraTreesRegressor(n_estimators=100).fit(X_train, y_train)
perm_etr = PermutationImportance(etr_model).fit(X_test, y_test)
eli5.show_weights(perm_etr, feature_names = X_test.columns.tolist())
# Feature importance for SGDRegressor
sgdr_model = SGDRegressor().fit(X_train, y_train)
perm_sgdr = PermutationImportance(sgdr_model).fit(X_test, y_test)
eli5.show_weights(perm_sgdr, feature_names = X_test.columns.tolist())