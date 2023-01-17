# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Files
import os

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print("Datas : ")
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train ", train.head())
print("Test ", test.head())
print("Train columns ", train.columns)
print("Test columns ", test.columns)
train.info()
print('_'*40)
test.info()
print('Train columns with null values:\n', train.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', test.isnull().sum())
print("-"*10)
train.head()
train = train.drop(columns=['vehicleDestroys', 'rideDistance', 'roadKills', 'matchId', 'swimDistance', 'walkDistance'])
test = test.drop(columns=['vehicleDestroys', 'rideDistance', 'roadKills', 'matchId', 'swimDistance', 'walkDistance'])

train.mean()
# Taking input data without the label
X_train = train.loc[:, train.columns != 'winPlacePerc']
# Label is the last column
y_train = train.iloc[:,-1]
X_test = test
y_train.head()
X_train.head()
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
y_test.head()
preds = reg.predict(X_test)
preds_train = reg.predict(X_train)

# print the coefficients
print(reg.intercept_)
print(reg.coef_)
# calculate MAE, MSE, RMSE
print(metrics.mean_absolute_error(y_train, preds_train))
print(metrics.mean_squared_error(y_train, preds_train))
print(np.sqrt(metrics.mean_squared_error(y_train, preds_train)))
result = pd.DataFrame(data=[X_test['Id'],preds]).T
result.Id = result.Id.astype(int)
print(X_test['Id'].head())
print(result)
result = result.rename(index=str, columns={"Unnamed 0": "winPlacePerc"})
result.columns
os.listdir("../input")
file = result.to_csv('PUBG_preds.csv', index=False)
print(file)
os.listdir()