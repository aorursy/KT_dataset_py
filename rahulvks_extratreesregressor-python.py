import pandas as pd

import numpy as np

import re

import string

import nltk

import os

import numpy as np

import pandas as pd

import xgboost as xgb

import gc

import seaborn as sns

import matplotlib.pyplot as plt







train = pd.read_csv('../input/train_2016.csv')

prop = pd.read_csv('../input/properties_2016.csv')

sample = pd.read_csv('../input/sample_submission.csv')



# Any results you write to the current directory are saved as output.
train.head(1)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
prop.head(1)
plt.figure(figsize=(8,6))

plt.scatter(range(train.shape[0]), np.sort(train.logerror.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()
for i, dtype in zip(prop.columns, prop.dtypes):

	if dtype == np.float64:

		prop[i] = prop[i].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.isnull().sum()
df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])

df_train["year"] = df_train["transactiondate"].dt.year

df_train["month"] = df_train["transactiondate"].dt.month

df_train["day"] = df_train["transactiondate"].dt.day
cnt_srs = df_train['month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8,palette='GnBu_d')

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()

cnt_srs = df_train['day'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8,palette='GnBu_d')

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
df_train=df_train.fillna(df_train.mean())
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplaceflag','hashottuborspa'], axis=1)

y_train = df_train['logerror'].values
from sklearn import preprocessing 

for f in x_train.columns: 

    if x_train[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder() 

        lbl.fit(list(x_train[f].values)) 

        x_train[f] = lbl.transform(list(x_train[f].values))
split = 80000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
print ('x_train',x_train.shape)

print ('y_train',y_train.shape)

print ('x_valid',x_valid.shape)

print ('y_valid',y_valid.shape)
from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=5, max_depth=10, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(x_train, y_train)
#plot the importances #

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(14,10))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices],

       color="b", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), indices)

plt.xlim([-1, len(indices)])

plt.show()
from sklearn import model_selection

seed = 7

kfold = model_selection.KFold(n_splits=10, random_state=seed)

model = ensemble.ExtraTreesRegressor(n_estimators=5, max_depth=10, max_features=0.3, n_jobs=-1, random_state=0)

results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)

print(results.mean())