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
!pip install fastai==0.7.0
!pip install torchtext==0.2.3
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from IPython.display import display

from sklearn import metrics
%matplotlib inline
df_raw = pd.read_csv('../input/train_V2.csv', low_memory=False)
df_raw_test = pd.read_csv('../input/test_V2.csv', low_memory=False)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
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
# m = RandomForestRegressor(n_jobs=-1)
# m.fit(df_trn, y_trn)
# m.score(df_trn,y_trn)
# split the data to train valid
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_trn, y_trn, test_size=0.2, random_state=42)
from sklearn.metrics import mean_absolute_error

def print_score(m):
    res = [mean_absolute_error(m.predict(X_train), y_train), mean_absolute_error(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
# m = RandomForestRegressor(n_jobs=-1)
# %time m.fit(X_train, y_train)
# print_score(m)
# df_train, y_train, nas = proc_df(df_raw, 'winPlacePerc', subset=100000, na_dict=nas)
# df_test, _, _ = proc_df(df_raw_test, na_dict=nas)

# X_train, _, y_train, _ = train_test_split(df_train, y_train, test_size=0.2, random_state=42)
# m = RandomForestRegressor(n_jobs=-1)
# %time m.fit(X_train, y_train)
# print_score(m)
# m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)
# # from IPython import IPython
# import graphviz

# draw_tree(m.estimators_[0], df_trn, precision=3)
# def draw_tree(t, df, size=10, ratio=0.6, precision=0):
#     """ Draws a representation of a random forest in IPython.

#     Parameters:
#     -----------
#     t: The tree you wish to draw
#     df: The data used to train the tree. This is used to get the names of the features.
#     """
#     s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
#                       special_characters=True, rotate=True, precision=precision)
#     IPython.display.display(graphviz.Source(re.sub('Tree {',
#        f'Tree {{ size={size}; ratio={ratio}', s)))
# m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)
# m = RandomForestRegressor(n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)
# preds = np.stack([t.predict(X_valid) for t in m.estimators_])
# preds[:,0], np.mean(preds[:,0]), y_valid[0]
# preds.shape
# plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);
# m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)
# m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)
# m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
# m.fit(X_train, y_train)
# print_score(m)
# m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
# m.fit(X_train, y_train)
# print_score(m)
from sklearn.model_selection import train_test_split

df_train, y_train, nas = proc_df(df_raw, 'winPlacePerc')
df_test, _, _ = proc_df(df_raw_test, na_dict=nas)

X_train, X_valid, y_train, y_valid = train_test_split(df_train, y_train, test_size=0.2, random_state=42)
set_rf_samples(50000)
m = RandomForestRegressor(n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
# reset_rf_samples()
# def dectree_max_depth(tree):
#     children_left = tree.children_left
#     children_right = tree.children_right

#     def walk(node_id):
#         if (children_left[node_id] != children_right[node_id]):
#             left_max = 1 + walk(children_left[node_id])
#             right_max = 1 + walk(children_right[node_id])
#             return max(left_max, right_max)
#         else: # leaf
#             return 1

#     root_node_id = 0
#     return walk(root_node_id)
# m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
# m.fit(X_train, y_train)
# print_score(m)
# t=m.estimators_[0].tree_
# dectree_max_depth(t)
# m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)
# m.fit(X_train, y_train)
# print_score(m)
# t=m.estimators_[0].tree_
# dectree_max_depth(t)

pred = m.predict(df_test)
pred
df_sub = df_raw_test_info[['Id']]
df_sub['winPlacePerc'] = pred
df_sub.to_csv('PUBG_sub.csv', index=None)
df_sub
