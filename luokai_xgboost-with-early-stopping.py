import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg") #Needed to save figures

from sklearn import cross_validation

import xgboost as xgb

from sklearn.metrics import roc_auc_score



training = pd.read_csv("../input/train.csv", index_col=0)

test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)

print(test.shape)
training = training.replace(-999999,2)
X = training.iloc[:,:-1]

y = training.TARGET
X['n0'] = (X == 0).sum(axis=1)
training.head()
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA
X.head()
X[:3].to_csv('a.csv')