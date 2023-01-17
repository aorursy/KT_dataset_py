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
from fastai.structured import *
from fastai import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.model_selection import *
!pip install ggplot
import pathlib
PATH = '/kaggle/input/ieee-fraud-detection/'
working_path = '/kaggle/working/'

path = pathlib.Path(PATH)
path_w = pathlib.Path(working_path)
!head -n 100000 {path}/train_transaction.csv > {path_w}/sample_train_transaction.csv
!head -n 100000 {path}/train_identity.csv > {path_w}/sample_train_identity.csv
trans = pd.read_csv(path_w/'sample_train_transaction.csv')
iden = pd.read_csv(path_w/'sample_train_identity.csv')
data = pd.merge(left=trans,right=iden,on='TransactionID',how='left')
obj_cols = data.dtypes[data.dtypes == object].index.tolist()
# obj_cols_df = pd.DataFrame({col:data[col].unique() for col in obj_cols})
for col in obj_cols:
    print(f'{col}\t{data[col].unique()}')
train_cats(data)
for col in obj_cols:
    print(f'{col}\t{data[col].unique()}')
def split_vals(a,n): return a[:n],a[n:]

df_trn, y_trn, nas = proc_df(data,y_fld='isFraud')

n_valid = 11810
n_train = len(df_trn) - n_valid

X_train, X_valid = split_vals(df_trn,n_train)
y_train, y_valid = split_vals(y_trn, n_train)
train_raw, valid_raw = split_vals(data, n_train)
set_rf_samples(50000)
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, roc_curve, auc
def print_score(m,imp_cols=None):
    if(imp_cols is not None):
        results = [roc_auc_score(y_train,m.predict(X_train[imp_cols])),roc_auc_score(y_valid,m.predict(X_valid[imp_cols])),
                   m.score(X_train[imp_cols],y_train),m.score(X_valid[imp_cols],y_valid)]
    else:
        results = [roc_auc_score(y_train,m.predict(X_train)),roc_auc_score(y_valid,m.predict(X_valid)),
                   m.score(X_train,y_train),m.score(X_valid,y_valid)]
        
    if(m.oob_score_):
        print(f'Score = {results}, OOB_SCORE = {m.oob_score_}')
    else:
        print("Score = ",results)
m = RandomForestRegressor(n_estimators=800,min_samples_leaf=3,max_features=1,n_jobs=-1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)
fi = rf_feat_importance(m,df_trn)
def plot_fi(fi): return fi.plot('cols','imp','barh',figsize=(12,10),legend=True)
plot_fi(fi[fi['imp'] > 0.003])
imps_cols = fi[fi['imp'] > 0.004]['cols'].tolist()
imps_cols
data[imps_cols].dtypes
len(imps_cols)
def split_vals(a,n): return a[:n],a[n:]

df_trn, y_trn, nas = proc_df(data,max_n_cat=10,y_fld='isFraud')

n_valid = 11810
n_train = len(df_trn) - n_valid

X_train, X_valid = split_vals(df_trn,n_train)
y_train, y_valid = split_vals(y_trn, n_train)
train_raw, valid_raw = split_vals(data, n_train)
X_train.columns.tolist()
m = RandomForestRegressor(n_estimators=800,min_samples_leaf=3,max_features=1,n_jobs=-1,oob_score=True)
m.fit(X_train[imps_cols],y_train)
print_score(m,imp_cols=imps_cols)
fi = rf_feat_importance(m,X_train[imps_cols])
def plot_fi(fi): return fi.plot('cols','imp','barh',figsize=(12,10),legend=True)
plot_fi(fi[:25])
imps_cols = fi[:25]['cols'].tolist()
data[imps_cols].dtypes
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(X_train[imps_cols]).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_train[imps_cols].columns, orientation='left', leaf_font_size=16)
plt.show()
def get_oob(df):
    m = RandomForestRegressor(n_estimators=800,min_samples_leaf=3,max_features=1,n_jobs=-1,oob_score=True)
    x,_ = split_vals(df,n_train)
    m.fit(x,y_train)
    return m.oob_score_
get_oob(X_train[imps_cols])
for col in ['C7','C12','V294','V317']:
    print(col,get_oob(X_train[imps_cols].drop(col,axis=1)))
to_drop = ['V294','V317']
get_oob(X_train[imps_cols].drop(to_drop,axis=1))
# to_drop = ['C7','C12','V294','V317']
# get_oob(X_train[imps_cols].drop(to_drop,axis=1))
m = RandomForestRegressor(n_estimators=800,min_samples_leaf=3,max_features=1,n_jobs=-1,oob_score=True)
cols = X_train[imps_cols].drop(to_drop,axis=1).columns
m.fit(X_train[cols],y_train)
print_score(m,imp_cols=cols)
