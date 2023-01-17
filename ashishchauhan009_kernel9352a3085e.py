# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn

from torch import optim

from torch.nn import functional as F

from matplotlib import pyplot as plt

import eli5

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

import sklearn.linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


df_train=pd.read_csv("../input/train.csv")

df_test =pd.read_csv("../input/test.csv")
print(df_train.shape, df_test.shape)
df_train.describe()
df_test.describe()
df_train1=df_train.drop(['target','id'], axis=1)
df_train1.describe()
df_test1=df_test.drop(['id'], axis=1)
df_test1.describe()
df_train1[df_train1.columns[2:]].std().plot('hist');

print(df_train1.shape)

plt.title('Distribution of stds');
df_train1.isnull().any().any()
df_train['target'].value_counts()
X_train = df_train1

y_train = df_train['target']

X_test = df_test1

n_fold = 20

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

repeated_folds = RepeatedStratifiedKFold(n_splits=20, n_repeats=20, random_state=42)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.feature_selection import RFE

logreg = LogisticRegression()

rfe = RFE(logreg, 20)

rfe = rfe.fit(X_train, y_train.values.ravel())

# print(rfe.support_)

print(rfe.ranking_)

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
y_pred = logreg.predict(X_test2)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test2, y_test2)))