!pip install fastai==0.7.0 
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.imports import *

from fastai.structured import *   ## Need to install fastai 0.7 for this 



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from IPython.display import display



from sklearn.model_selection import train_test_split

from sklearn import metrics



import os

print(os.listdir("../input"))
df_raw = pd.read_csv('../input/train_V2.csv', low_memory=False)

df_raw_test = pd.read_csv('../input/test_V2.csv', low_memory=False)
def display_all(df):

    with pd.option_context("display.max_rows", 100, "display.max_columns", 100):

        display(df)
display_all(df_raw.tail())
display_all(df_raw.describe(include='all'))
# store test info

df_raw_test_info = df_raw_test[['Id', 'groupId', 'matchId']]
df_raw.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)

df_raw_test.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
train_cats(df_raw)

apply_cats(df_raw_test, df_raw)
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
df_raw[pd.isna(df_raw['winPlacePerc'])]
df_raw.dropna(subset=['winPlacePerc'], inplace=True)
df_trn, y_trn, nas = proc_df(df_raw, 'winPlacePerc')

df_test, _, _ = proc_df(df_raw_test, na_dict=nas)
# split the data to train valid



X_train, X_valid, y_train, y_valid = train_test_split(df_trn, y_trn, test_size=0.33, random_state=42)
from sklearn.metrics import mean_absolute_error



def print_score(m):

    res = [mean_absolute_error(m.predict(X_train), y_train), mean_absolute_error(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
set_rf_samples(20000)

m = RandomForestRegressor(n_jobs=-1, n_estimators = 40, min_samples_leaf = 7, min_samples_split = 7)

%time m.fit(X_train, y_train)
%time print_score(m)
pred = m.predict(df_test)

pred
df_sub = df_raw_test_info[['Id']]
pd.options.mode.chained_assignment = None  # default='warn' ## TO remove warning due to assignment below

df_sub['winPlacePerc'] = pred
df_sub.to_csv('PUBG_RF_tune.csv', index=None)
fi = rf_feat_importance(m, X_train)

fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False)
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30])
to_keep = fi[fi.imp>0.005].cols; len(to_keep)
df_keep = df_raw[to_keep].copy()

X_train, X_valid, y_train, y_valid = train_test_split(df_keep, y_trn, test_size=0.33, random_state=42)
set_rf_samples(20000)

m = RandomForestRegressor(n_jobs=-1, n_estimators = 40, min_samples_leaf = 7, min_samples_split = 7)

%time m.fit(X_train, y_train)
%time print_score(m)