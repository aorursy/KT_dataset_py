# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing data

complete_bottle_data = pd.read_csv("../input/calcofi/bottle.csv")

partial_bottle_data = complete_bottle_data.loc[1:1000, ['T_degC','Salnty', 'Depthm']]

partial_bottle_data.head()
partial_bottle_data.describe()
# Creating train and test dataset

msk = np.random.rand(len(partial_bottle_data)) < 0.8

train_raw = partial_bottle_data[msk]

test_raw = partial_bottle_data[~msk]
# Fixing missing values issue and imputing

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

train = pd.DataFrame(imputer.fit_transform(train_raw))

train.columns = partial_bottle_data.columns

train.rename(columns={'T_degC':'TEMP', 'Salnty':'SALINITY', 'Depthm':'DEPTH'}, inplace=True)

train = train.reindex(columns={'SALINITY','DEPTH','TEMP'})

train.head()
train.describe()
# Modelling

from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['SALINITY', 'DEPTH']])

train_y = np.asanyarray(train[['TEMP']])

regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
# Predicting begins

# Applying same Fixing missing values issue and imputing

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

test = pd.DataFrame(imputer.fit_transform(test_raw))

test.columns = partial_bottle_data.columns

test.rename(columns={'T_degC':'TEMP', 'Salnty':'SALINITY', 'Depthm':'DEPTH'}, inplace=True)

test = test.reindex(columns={'SALINITY','DEPTH', 'TEMP'})

test.shape
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['SALINITY','DEPTH']])

test_y = np.asanyarray(test[['TEMP']])

test_y_hat = regr.predict(test_x)





print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )