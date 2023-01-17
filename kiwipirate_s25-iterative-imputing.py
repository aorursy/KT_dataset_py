# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
stations = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S43', 'S44', 'S45', 'S47', 'S48', 'S49', 'S50','S51']

import sklearn

import sys

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn import neighbors

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.impute import KNNImputer

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDRegressor

import gc

estimators = [

    DecisionTreeRegressor(max_features='sqrt', random_state=0),

    ExtraTreesRegressor(n_estimators=10, random_state=0),

    KNeighborsRegressor(n_neighbors=17),

    RandomForestRegressor(n_estimators = 10,random_state=1),

    SGDRegressor(max_iter=1000, tol=1e-3)

]





df = pd.read_csv('/kaggle/input/bosch-numeric/S25_numeric.csv')

clone = df.copy()

index = pd.isnull(clone).any(1).to_numpy().nonzero()

print(index)
%%time

my_imputer = IterativeImputer(estimator =estimators[4] ,random_state = 10  ,tol = 0.001 ,sample_posterior =False)

my_imputer.fit(clone)

imputed_data1 = pd.DataFrame(data = my_imputer.transform(clone),columns = clone.columns)
imputed_data1.iloc[index[0],:]