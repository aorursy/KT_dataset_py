# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print('Input directory listing:\n', os.listdir("../input"))



# Any results you write to the current directory are saved as output.



data = pd.read_csv('../input/train.csv')

print(data.columns)

print(data.dtypes)

print(data.head())

print(data.tail())

print(data.describe())
# perm imp for hosp readmit



# def target, features

y = data['readmitted']

X = data.loc[:, :'diabetesMed_Yes']

print('-'*74)

print('X: ', X.columns)

print('-'*74)



# train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
# perm imp

import eli5

from eli5.sklearn import PermutationImportance



perm_imp = PermutationImportance(model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm_imp, feature_names = val_X.columns.tolist())