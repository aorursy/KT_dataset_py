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
# In teaching myself how to use ensemble learning, I came across this version of the Wisconsin cancer data and wanted to apply the tutorial I was using to see how it scored.

# Tutorial can be seen here https://tinyurl.com/y49hz66t
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier
url = '../input/data.csv'

df = pd.read_csv(url, index_col = 0)

df.head(3)
df.replace('?',0,inplace=True)

df.diagnosis = pd.get_dummies(df.diagnosis)

df = df.drop(labels='Unnamed: 32',axis=1)

scaler = MinMaxScaler(feature_range=(0,1))

normalizedData = scaler.fit_transform(df.values)

names = list(df.columns)
normalizedData = pd.DataFrame(normalizedData)

normalizedData.columns = names

x = normalizedData.loc[:,normalizedData.columns != 'diagnosis']

y = normalizedData.diagnosis
kfold = model_selection.KFold(n_splits=10, random_state=17)

cart = DecisionTreeClassifier()

num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=17)

results = model_selection.cross_val_score(model, x, y, cv=kfold)

print(results.mean())
from sklearn.ensemble import AdaBoostClassifier

seed = 17

num_trees = 100

kfold = model_selection.KFold(n_splits=10, random_state=seed)

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

results = model_selection.cross_val_score(model, x, y, cv=kfold)

print(results.mean())