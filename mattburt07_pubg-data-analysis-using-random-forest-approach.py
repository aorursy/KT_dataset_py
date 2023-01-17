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
df_raw = pd.read_csv('../input/train_V2.csv')
df_raw.shape
df_raw.head()
df_raw.isnull().sum()
df_raw = df_raw.dropna()
df_raw.shape
df_raw.columns
df_raw['totalDistance'] = df_raw['rideDistance'] + df_raw['walkDistance'] + df_raw['swimDistance']
ranking_cats = ['killPlace', 'damageDealt', 'kills', 'walkDistance', 'rankPoints', 'weaponsAcquired', 'totalDistance']
for c in ranking_cats: df_raw[c+'_ranking'] = df_raw.groupby('matchId')[c].rank(ascending=False)
df_raw.head()
df_raw = df_raw.sort_values(['matchId'])
train_cats(df_raw)
df, y, nas = proc_df(df_raw, 'winPlacePerc', max_n_cat=5)
df.shape
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 444647  #around 20% of the training data
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
set_rf_samples(100000)
m = RandomForestRegressor(n_estimators=100, max_features=0.5, min_samples_leaf=3, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df); fi[:]
fi.plot('cols', 'imp', figsize=(12,6), legend=False, xticks=np.arange(37));
plt.xticks(rotation=90)
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:]);
to_keep = fi[fi.imp>0.0025].cols; len(to_keep)
df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=100, max_features=0.5, min_samples_leaf=3, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
drop_cols = ['totalDistance', 'walkDistance', 'groupId', 'matchId', 'Id']
df_final = df_keep.drop(drop_cols, axis=1)
np.save('keep_cols.npy', np.array(df_final.columns))
keep_cols = np.load('keep_cols.npy')
df_keep = df[keep_cols]
X_train, X_valid = split_vals(df_keep, n_trn)
reset_rf_samples()
m = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
df_test = pd.read_csv('../input/test_V2.csv')
df_test['totalDistance'] = df_test['rideDistance'] + df_test['walkDistance'] + df_test['swimDistance']
ranking_cats = ['killPlace', 'damageDealt', 'kills', 'walkDistance', 'rankPoints', 'weaponsAcquired', 'totalDistance']
for c in ranking_cats: df_test[c+'_ranking'] = df_test.groupby('matchId')[c].rank(ascending=False)
df_test.head()
train_cats(df_test)
df_test, y, nas = proc_df(df=df_test, y_fld=None)
df_submit = df_test[keep_cols]
a = m.predict(df_submit)
a = pd.Series(a)
submission = pd.Series((pd.read_csv('../input/test_V2.csv', low_memory=False))['Id'])
submission = pd.concat([submission, a], axis=1)
submission = submission.rename(columns={0:'winPlacePerc'})
submission.to_csv('submission.csv', index=False)