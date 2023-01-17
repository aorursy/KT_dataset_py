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

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

train.shape
train.describe()
train.info()
train.info()
test = pd.read_csv('../input/test.csv')

test.shape
train_X = train.drop('SalePrice' , axis =1)

train_y = train['SalePrice']

test_X = test

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer
onehot_train_X = pd.get_dummies(train_X)

onehot_test_X = pd.get_dummies(test_X)

train_X, test_X = onehot_train_X.align(onehot_test_X, join='left', axis=1)

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)
reg = LinearRegression()

cv_scores = cross_val_score(reg, train_X, train_y, cv=5)

print(cv_scores)

reg.fit(train_X, train_y)

predictions = reg.predict(test_X)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice':predictions})

my_submission.to_csv('submission.csv', index=False)