# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



from numba import jit

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold

import seaborn as sns



import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



print("Setup ok!")
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
test_df.shape, train_df.shape
train_df.head()
test_df.head()
train_df.isnull().values.any()
test_df.isnull().values.any()