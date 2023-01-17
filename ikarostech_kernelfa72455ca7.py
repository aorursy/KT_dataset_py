import os; os.environ['OMP_NUM_THREADS'] = '1'

from contextlib import contextmanager

from functools import partial

from operator import itemgetter

from multiprocessing.pool import ThreadPool

import time

from typing import List, Dict



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='white', context='notebook', palette='Set2')



import keras as ks

import pandas as pd

import numpy as np

import tensorflow as tf

#from sklearn import preprocessing

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf

from sklearn.pipeline import make_pipeline, make_union, Pipeline

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score, train_test_split



import lightgbm as lgb



import sys

import gc

path_train = '../input/train_V2.csv'

path_test = '../input/test_V2.csv'
train = pd.read_csv(path_train)

test = pd.read_csv(path_test)

train.head()

train.describe()
test.head()
feats = [f for f in train.columns if f not in ["Id","groupId","matchId"]]



plt.figure(figsize=(18,16))

sns.heatmap(train[feats].corr(), linewidths=0.1,vmax=1.0,

               square=True, linecolor='white', annot=True, cmap="RdBu")
plt.figure(figsize=(12,6))

sns.distplot(train["winPlacePerc"].values,bins=100,kde=False)
#preprocess_data

train = train.drop(["Id","groupId","matchId","matchType"],1)

evaliation = test.drop(["Id","groupId","matchId","matchType"],1)



train_df = train.copy()

full_X = train_df.drop(["winPlacePerc"],1)

full_Y = train_df["winPlacePerc"]
train_X,valid_X,train_Y,valid_Y = train_test_split(full_X,full_Y,test_size=0.2,random_state=42)

train_X.head()
train_data = lgb.Dataset(train_X,label=train_Y)

eval_data = lgb.Dataset(valid_X,label=valid_Y,reference=train_data)
params = {

    'boosting_type':'gbdt',

    'objective':'regression',

    'metric':'l2',

    'num_leaves': 144,

    'learning_rate':0.05,

    'feature_fraction': 0.9,                                                                             

    'bagging_fraction': 0.8,                                                                             

    'bagging_freq': 5,   

    'n_estimators': 800,

    'max_depth':12,

    'max_bin':55,

    'verbose':50, 

    'lambda_l2': 2,

}

gbm = lgb.train(params,

               train_data,

               num_boost_round=200,

               valid_sets=eval_data,

               early_stopping_rounds=10

               )
pred = gbm.predict(evaliation,num_iteration=gbm.best_iteration)
wPP = pd.Series(data=pred,name="winPlacePerc")

result = pd.DataFrame(index=[test["Id"],wPP])

#result['winPlacePerc'] = pred
result.to_csv("submission.csv",index=False)