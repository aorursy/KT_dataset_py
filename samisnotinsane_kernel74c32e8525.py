# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install fastai==0.7.0
from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics
PATH = "/kaggle/input/titanic/"

!ls {PATH}
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False)
df_raw.tail()
df_raw.describe(include='all')
m = RandomForestRegressor(n_jobs=-1)

m.fit(df_raw.drop('Survived', axis=1), df_raw.Survived)
train_cats(df_raw)
df_raw.Name.cat.categories
df_raw.Name = df_raw.Name.cat.codes
m = RandomForestRegressor(n_jobs=-1)

m.fit(df_raw.drop('Survived', axis=1), df_raw.Survived)
df_raw.tail()
df_raw.Age = df_raw.Age.cat.categories
train_cats(df_raw)
df_raw.Age = df_raw.Age.cat.categories
df, y, nas = proc_df(df_raw, 'Survived')
m = RandomForestRegressor(n_jobs=-1)

m.fit(df, y)

m.score(df, y)
def split_vals(a, n): return a[:n].copy(), a[n:].copy()



n_valid = 418 # how can I find out the test set size? Our total no. of records = 891. test.csv has 418 records, so using that value

n_trn = len(df) - n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



X_train.shape, y_train.shape, X_valid.shape
def rmse(x, y): return math.sqrt( ((x-y)**2).mean() ) 
def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid), m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'):

        res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
df_trn, y_trn, nas = proc_df(df_raw, 'Survived', subset=30000, na_dict=nas)

X_train, _ = split_vals(df_trn, 20000)

y_train, _ = split_vals(y_trn, 20000)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
predictions = m.predict(X_train)
submission = pd.DataFrame({'PassengerId':X_train['PassengerId'],'Survived':predictions})

submission.head()
df_raw.head()
filename = 'Titanic-Predictions-1.csv'

submission.to_csv(filename, index=False)

print('Saved file: ' + filename)
from IPython.display import FileLink

FileLink(r'Titanic-Predictions-1.csv')