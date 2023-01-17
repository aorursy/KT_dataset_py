# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/sudeste.csv',low_memory=False,parse_dates=['mdct'],nrows=100000)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
    
display_all(data.head(100).T)
add_datepart(data, 'mdct')
data.head(100)
train_cats(data)
data.wsnm.cat.categories
data.wsnm = data.wsnm.cat.codes
display_all(data.head().T)
data.prov.cat.categories
data = data.drop(['city','date'],axis=1)
data.dtypes
display_all(data.isnull().sum().sort_index()/len(data))
df, y, nas = proc_df(data, 'temp')
display_all(df.head())
df
learn = RandomForestRegressor(n_jobs=-1)
learn.fit(df,y)
learn.score(df,y)
def split_vals(a,n): return a[:n].copy(),a[n:].copy()
valid = 20000
train = len(df) - valid
x_train,x_valid = split_vals(df,train)
y_train,y_valid = split_vals(y,train)

x_train.shape,y_train.shape,x_valid.shape


learn.fit(x_train,y_train)
learn.score(x_valid,y_valid)
learn = RandomForestRegressor(n_estimators=40,n_jobs=-1)
learn.fit(x_train,y_train)
learn.score(x_train,y_train),learn.score(x_valid,y_valid)

