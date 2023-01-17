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
%load_ext autoreload
%autoreload 2

%matplotlib inline
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
train = pd.read_csv('../input/train.csv')
train.head()
train.shape
train.info()
train.describe(include='all').T
train.SalePrice = np.log(train.SalePrice)
train_cats(train)
# ??train_cats
train.Street.cat.categories, train.Street.cat.codes.head()
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(train.isnull().sum().sort_index()/len(train))
os.makedirs('tmp', exist_ok=True)
train.to_feather('tmp/bulldozers-raw')
df_raw = pd.read_feather('tmp/bulldozers-raw')
df, y, nas = proc_df(df_raw, 'SalePrice')
df.head()
m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df,y)
test_data = pd.read_csv('../input/test.csv')
test_data.shape
apply_cats(test_data, train)
test_data['SalePrice'] = 0
test_data.head()
tt, _, nas1 = proc_df(test_data,'SalePrice',na_dict=nas)
nas1
tt.shape
pp = m.predict(tt)
submission = pd.read_csv('../input/sample_submission.csv')
submission.head(1)
submission['SalePrice'] = np.exp(pp)
submission.head()
submission.to_csv('submission.csv', index=False)
