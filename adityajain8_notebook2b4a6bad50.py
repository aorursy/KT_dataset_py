# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
train = train.drop(columns=['id'])
# train[train['target'] == 1].count()
# data.isnull().sum().sum()  
# just to make sure there isn't any null entries in data
# X = train.iloc[:, :-1]
# y = train.iloc[:, 88]
# len(X)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.34)
# print(len(X_train), len(X_test), len(y_train))
# zeros = train[train['target'] == 0]
ones = train[train['target'] == 1]

for _ in range(5):
    ones = ones.append(ones, ignore_index = True)
#     print(len(ones))

train = train.append(ones, ignore_index = True)
train = train.sample(frac = 1)
y_train = train.iloc[:, 88]
X_train = train.iloc[:, :-1]
X_train
# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
# DT = DecisionTreeClassifier()

# scores = cross_validate(DT, X_train, y_train, scoring=scoring, cv=20)

# sorted(scores.keys())
# accuracy = scores['test_accuracy'].mean()
# roc = scores['test_roc_auc'].mean()
# accuracy


regres = LinearRegression()
regres.fit(X_train, y_train)
test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
test = test.drop(columns=['id'])
len(test)
# threshold = 0.008
y_pred = regres.predict(test)
# y_pred = y_pred < threshold
# print(y_pred.sum())
# print(y_test.sum())
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# roc_auc_score(y_test, y_pred)
y_pred
res = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
res = res['id']
res = pd.DataFrame(res)
len(res)
res['target'] = y_pred
# res['target'] = res['target'].apply(lambda x: 0 if x < 0.009 else 1)
res['target'] = res['target'].abs()
res.to_csv('/kaggle/working/submission_prob.csv', index = False)
