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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

import os
def num_missing(data):

    ms_counts = data.isnull().sum()

    return ms_counts[ms_counts != 0]
raw_train = pd.read_csv('../input/train.csv')

raw_test = pd.read_csv('../input/test.csv')
raw_train.head()
raw_test.head()
raw_train.info()
missing= num_missing(raw_train)

missing
raw_train[missing.index.values].dtypes
train = raw_train.copy()

test = raw_test.copy()
train['MSSubClass'] = train['MSSubClass'].astype('object')
cat_columns = train.select_dtypes('object').columns
train[cat_columns] = train[cat_columns].fillna('missing')
test[cat_columns] = test[cat_columns].fillna('missing')
train = pd.get_dummies(train)

test = pd.get_dummies(test)

train, test = train.align(test, join ='left', axis=1)
train_X = train.drop(columns=['Id', 'SalePrice'])

train_y =train.SalePrice

test_X = test.drop(columns =['Id', 'SalePrice'])
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
model = Pipeline([

    ('scale',StandardScaler()),

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('reg', XGBRegressor(objective='reg:squarederror',n_estimators=1000))

])
from sklearn.model_selection import cross_val_score



cross_val_score(model, train_X,train_y, cv=5)
model.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error



preds = model.predict(train_X)

mean_absolute_error(preds, train_y)
test_preds = model.predict(test_X)



output = pd.DataFrame({'Id':test.Id,'SalePrice':test_preds})

output.to_csv('submission.csv',index=False)