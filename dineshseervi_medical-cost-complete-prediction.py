#loading_all libraries

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pylab as pylab

from pandas import get_dummies

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import sys

import csv

import os
warnings.filterwarnings('ignore')

%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train=pd.read_csv('/kaggle/input/insurance/insurance.csv')
df_train.head()
#function for missing data

def missing_data(df_train):

    total = df_train.isnull().sum().sort_values(ascending=False)

    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return(missing_data.head(20))
missing_data(df_train)
df_train['children']=df_train['children'].astype('object')
df_train.dtypes
encoded=pd.get_dummies(df_train)
encoded.head()
dependent_all=encoded['charges']

independent_all=encoded.drop(['charges'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(independent_all,dependent_all,test_size=0.3,random_state=100)
linregr = LinearRegression()

linregr.fit(x_train, y_train)

pred_linreg = linregr.predict(x_train)
print("accuracy score for train using linearregression is",linregr.score(x_train,y_train))

print("accuracy score for test using linearregression is",linregr.score(x_test,y_test))
mae_train = mean_absolute_error(linregr.predict(x_train),y_train)

print('Mae on train using linearregression :',mae_train)

mae_test = mean_absolute_error(linregr.predict(x_test),y_test)

print('Mae on test using linearregression :',mae_test)



mse_train = mean_squared_error(linregr.predict(x_train),y_train)

print('Mse on train using linearregression :',mse_train)

mse_test = mean_squared_error(linregr.predict(x_test),y_test)

print('Mse on test using linearregression :',mse_test)
#random Forest

rfr = RandomForestRegressor(n_estimators=100)

rfr.fit(x_train,y_train)

predicted = rfr.predict(x_train)
print("accuracy score for train using randomforest",rfr.score(x_train,y_train))

print("accuracy score for test using randomforest",rfr.score(x_test,y_test))
mae_train = mean_absolute_error(rfr.predict(x_train),y_train)

print('Mae on train using Randomforest :',mae_train)

mae_test = mean_absolute_error(rfr.predict(x_test),y_test)

print('Mae on test using Randomforest :',mae_test)



mse_train = mean_squared_error(rfr.predict(x_train),y_train)

print('Mse on train using Randomforest :',mse_train)

mse_test = mean_squared_error(rfr.predict(x_test),y_test)

print('Mse on test using Randomforest :',mse_test)
#decisiontreeregressor

dtr = DecisionTreeRegressor()

dtr.fit(x_train,y_train)

predicted=dtr.predict(x_train)
print("accuracy score for train using Decision_tree",dtr.score(x_train,y_train))

print("accuracy score for test using Decision_tree",dtr.score(x_test,y_test))
mae_train = mean_absolute_error(dtr.predict(x_train),y_train)

print('Mae on train using decision tree :',mae_train)

mae_test = mean_absolute_error(dtr.predict(x_test),y_test)

print('Mae on test using decision tree :',mae_test)



mse_train = mean_squared_error(dtr.predict(x_train),y_train)

print('Mse on train using decision tree :',mse_train)

mse_test = mean_squared_error(dtr.predict(x_test),y_test)

print('Mse on test using decision tree :',mse_test)
xgboost = xgb.XGBRegressor(n_estimators=300)

xgboost.fit(x_train,y_train)

predicted=xgboost.predict(x_train)
print("accuracy score for train using Xgboost:",xgboost.score(x_train,y_train))

print("accuracy score for test using Xgboost :",xgboost.score(x_test,y_test))
mae_train = mean_absolute_error(xgboost.predict(x_train),y_train)

print('Mae on train using XGboost :',mae_train)

mae_test = mean_absolute_error(xgboost.predict(x_test),y_test)

print('Mae on test using XGboost :',mae_test)



mse_train = mean_squared_error(xgboost.predict(x_train),y_train)

print('Mse on train using XGboost :',mse_train)

mse_test = mean_squared_error(xgboost.predict(x_test),y_test)

print('Mse on test using XGboost :',mse_test)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(x_train,y_train)
print("accuracy score for train using randomforest gridsearch:",rf_random.score(x_train,y_train))

print("accuracy score for test using randomforest gridsearch :",rf_random.score(x_test,y_test))