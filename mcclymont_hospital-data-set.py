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




# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline
!pip install fastai==0.7.0
from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import csv

from sklearn import metrics
path = '../input'
!ls {path}
df_raw = pd.read_csv(f'{path}/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv', low_memory=False)
df_raw.head()
print("n of columns: ", len(df_raw.columns))

df_raw.columns
len(df_raw)
train_cats(df_raw)
train_df, y, nas = proc_df(df_raw, 'Total Costs')
for n in y:

    print(n)
train_df = train_df.drop('Total Charges', axis=1)
for n in train_df.columns:

    print(n, train_df[n].unique())
def split_vals(a, n):

    return a[:n].copy(), a[n:].copy()
n_valid = round(len(df_raw) * 0.3)

n_trn = len(df_raw) - n_valid

n_trn, n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(train_df, n_trn)

y_train, y_valid = split_vals(y, n_trn)
def rmse(x, y):

    return math.sqrt(((x-y)**2).mean())
def print_score(m):

    res = [

        rmse(m.predict(X_train), y_train),

        rmse(m.predict(X_valid), y_valid),

        m.score(X_train, y_train),

        m.score(X_valid, y_valid)

    ]

    print(res)
m = RandomForestRegressor(n_jobs=-1, n_estimators=10, max_features=0.5)

%time m.fit(X_train, y_train)

print_score(m)
preds =  np.stack([t.predict(X_valid) for t in m.estimators_])
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0))

         for i in range(20)])
fi = rf_feat_importance(m, train_df)
fi
def plot_fi(fi):

    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi)
for n in df_raw['APR DRG Code'].unique():

    print(n)
x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)

x['pred'] = np.mean(preds, axis=0)

plotter = x['APR DRG Code'].value_counts().plot.barh(figsize=(15,50))

#this relates to diagnosis - but these codes are old so they are hard to find 
flds = ['APR DRG Code', 'Total Costs', 'pred', 'pred_std']

natureSumm = x[flds].groupby('APR DRG Code', as_index=False).mean().sort_values('Total Costs')

natureSumm


natureSumm = natureSumm[~pd.isnull(natureSumm['Total Costs'])]

natureSumm.plot('APR DRG Code', 'Total Costs', 'barh', figsize=(20,40))