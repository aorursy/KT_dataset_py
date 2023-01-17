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
%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import math

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

sns.set()
# display function from FAST AI

def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
PATH = "/kaggle/input/pubg-finish-placement-prediction/"
df_raw = pd.read_csv(F"{PATH}train_V2.csv")

df_test = pd.read_csv(F"{PATH}test_V2.csv")
display_all(df_raw.tail().T)
df_raw.describe(include='all').T
df_train = df_raw.copy()



# One hot encode matchType

df_train["matchType"] = pd.Categorical(df_train["matchType"])

dfDummies = pd.get_dummies(df_train['matchType'], prefix = 'match_type')

df_train = pd.concat([df_train, dfDummies], axis=1)

df_train = df_train.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1)

# One row is bugged, with a match not starting

df_train = df_train.dropna()
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = len(df_test)

n_trn = len(df_train) - n_valid



raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df_train.drop('winPlacePerc', axis=1), n_trn)

y_train, y_valid = split_vals(df_train.winPlacePerc, n_trn)



X_train.shape, y_train.shape, X_valid.shape
# Functions from FAST AI



def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [F"RMSE TRAIN: {rmse(m.predict(X_train), y_train)}",

           F"RMSE VALID: {rmse(m.predict(X_valid), y_valid)}",

                F"SCORE train: {m.score(X_train, y_train)}",

           F"SCORE VALID {m.score(X_valid, y_valid)}"]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.5)

%time m.fit(X_train, y_train)

print_score(m)
df_test_predict = df_test.copy()

df_test_predict["matchType"] = pd.Categorical(df_test_predict["matchType"])

dfDummies = pd.get_dummies(df_test_predict['matchType'], prefix = 'match_type')

df_test_predict = pd.concat([df_test_predict, dfDummies], axis=1)
predicted = m.predict(df_test_predict.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1))
df_test_predict["winPlacePerc"] = predicted
df_test_predict.loc[:,['Id', 'winPlacePerc']].to_csv("submission.csv", index=False)