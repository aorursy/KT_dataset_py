# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install fastai==0.7.0

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.imports import *

from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_raw = pd.read_csv('../input/train/Train.csv', low_memory=False, parse_dates=["saledate"])

df_test = pd.read_csv('../input/Test.csv',low_memory=False, parse_dates=["saledate"])

df_test.columns
#Converting saleprice to log since competition rules state using rmsle

df_raw.SalePrice = np.log(df_raw.SalePrice)
add_datepart(df_raw,'saledate')

add_datepart(df_test,'saledate')
train_cats(df_raw)

apply_cats(df_test,df_raw)
X, y , nas = proc_df(df_raw, 'SalePrice') #training

X_test, _, nas = proc_df(df_test, na_dict=nas)

X, y , nas = proc_df(df_raw, 'SalePrice', na_dict=nas)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_valid = 12000 #kaggle's test set size

n_trn = len(X) - n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(X, n_trn)  #splitting the data except the prediction variable 

y_train, y_valid = split_vals(y, n_trn)   #splitting the prediction variable - saleprice 
#Defining some functions for measuring performance 

def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.7, n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
prediction = m.predict(X_test)
submission = pd.DataFrame()

submission['id_column']=df_test.SalesID

submission['SalePrice']= prediction

submission.to_csv('submission.csv',index=False)