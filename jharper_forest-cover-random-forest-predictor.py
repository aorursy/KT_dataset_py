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
from fastai.imports import *
from fastai.structured import *
from fastai.plots import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

from sklearn import metrics
PATH = "../input/"
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False)
df, y, nas = proc_df(df_raw, 'Cover_Type')
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = int(len(df)*0)
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape, y_train.shape, X_valid.shape
def print_score(m):
    #res = [m.score(X_train, y_train), m.score(X_valid, y_valid)]
    res = [m.score(X_train, y_train)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestClassifier(n_estimators=50,  max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df);
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
to_keep = (fi[(fi.imp>0.001)]).cols; len(to_keep)
to_keep
to_drop = ['Id']
df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
def get_oob(df):
    m = RandomForestClassifier(n_estimators=50, max_features=0.5, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_
get_oob(df_keep)
to_drop = ['Id']
get_oob(df_keep.drop(to_drop, axis=1))
df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)
get_oob(df_keep)
m = RandomForestClassifier(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
df_test_raw = pd.read_csv(f'{PATH}test.csv', low_memory=False)
df_test_keep = df_test_raw[to_keep].copy()
df_test_keep.drop(to_drop, axis=1, inplace=True)
df_test_keep.shape
y_test_pred = m.predict(df_test_keep)
y_test_pred.shape
np.bincount(y_test_pred)
sub = df_test_raw[['Id']].copy()
sub['Cover_Type'] = y_test_pred
sub[0:20]
sub.to_csv("submission_0.csv", index=False)
