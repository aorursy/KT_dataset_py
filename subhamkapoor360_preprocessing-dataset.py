# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import glob

import operator

import pickle

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score,roc_curve

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
TARGET = 'target'
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

train.head(3)
print("Shape of train data: {}".format(train.shape))

print("Shape of test data: {}".format(test.shape))
train.drop('id',axis=1,inplace = True)

test.drop('id',axis=1,inplace = True)
target_distribution = train.target.value_counts()

print(target_distribution)

target_distribution.plot('bar')
target,features = train.target,train.drop('target',axis = 1)
def variable_uniqueness_rate(df):

	uniquness_rate = {}

	for col in df.columns:

		rate = df[col].value_counts(normalize=True, dropna=False).values[0]

		uniquness_rate.update({col:rate*100})

	uniquness_rate = sorted(uniquness_rate.items(), key=operator.itemgetter(1),reverse = True)

	return uniquness_rate



def get_feature_unique(df):

	nunique_map = {}

	for col in df.columns:

		nunique_map.update({col:df[col].nunique()})

	nunique_map = sorted(nunique_map.items(), key=operator.itemgetter(1),reverse = True)

	return nunique_map



def get_missing_data_table(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data.to_dict()
variable_unique_rate = variable_uniqueness_rate(features)

unique_features = get_feature_unique(features)
unique_features
variable_unique_rate
train_dummy = pd.get_dummies(features['wheezy-copper-turtle-magic'], prefix= 'whz')

test_dummy = pd.get_dummies(test['wheezy-copper-turtle-magic'], prefix= 'whz')
features.drop('wheezy-copper-turtle-magic',axis = 1,inplace = True)

test.drop('wheezy-copper-turtle-magic',axis = 1,inplace = True)
features = pd.merge(features,train_dummy,left_index=True,right_index=True)

test = pd.merge(test,test_dummy,left_index=True,right_index=True)

print("Shape of train data: {}".format(features.shape))

print("Shape of test data: {}".format(test.shape))