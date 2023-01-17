# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
import pandas as pd

import os

import tarfile

from six.moves import urllib



sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

housing = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

data_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
housing.drop(['Id'], axis=1, inplace=True)
housing.head()
housing.info()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))

plt.show()


corr_matrix = housing.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)
housing = housing[housing.GrLivArea < 4500]

housing.reset_index(drop=True, inplace=True)
housing = housing[housing.SalePrice < 550000]

housing.reset_index(drop=True, inplace=True)
from sklearn import preprocessing

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import numpy as np
cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

x = housing[cols].values

y = housing['SalePrice'].values

x_scaled = preprocessing.StandardScaler().fit_transform(x)

y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))

X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
clfs = {

        'svm':svm.SVR(), 

        'RandomForestRegressor':RandomForestRegressor(n_estimators=400),

        'BayesianRidge':linear_model.BayesianRidge()

       }

for clf in clfs:

    try:

        clfs[clf].fit(X_train, y_train)

        y_pred = clfs[clf].predict(X_test)

        print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )

    except Exception as e:

        print(clf + " Error:")

        print(str(e))


from sklearn.ensemble import RandomForestRegressor

rfr = clf
data_test[cols].isnull().sum()
cols2 = ['OverallQual','GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

cars = data_test['GarageCars'].fillna(1.766118)

bsmt = data_test['TotalBsmtSF'].fillna(1046.117970)

data_test_x = pd.concat( [data_test[cols2], cars, bsmt] ,axis=1)

data_test_x.isnull().sum()
data_test_x.info()
x = data_test_x.values

y_te_pred = rfr.predict(data_test_x)

print(y_te_pred)



print(y_te_pred.shape)

print(x.shape)