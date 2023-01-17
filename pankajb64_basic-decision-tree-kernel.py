# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Source - blog.socialcops.com/engineering/machine-learning-python/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import random
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_df = pd.read_csv('../input/train.csv', index_col=None, na_values=['NA'])
titanic_df.head()
titanic_df.count()
titanic_df = titanic_df.drop(['Cabin'], axis=1)
titanic_df['Embarked'] = titanic_df['Embarked'].fillna("NA")
titanic_df = titanic_df.dropna()
titanic_df.count()
def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name', 'Ticket'], axis=1)
    return processed_df
processed_df = preprocess_titanic_df(titanic_df)
X = processed_df.drop(['Survived'], axis=1).values
y = processed_df['Survived'].values
clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit(X,y)
titanic_test = pd.read_csv('../input/test.csv', index_col=None, na_values=['NA'])
titanic_test = titanic_test.drop(['Cabin'], axis=1)
titanic_test['Embarked'] = titanic_test['Embarked'].fillna("NA")
titanic_test['Age'] = titanic_test['Age'].fillna(-1)
titanic_test['Fare'] = titanic_test['Fare'].fillna(-1)
processed_test = preprocess_titanic_df(titanic_test)
processed_test[processed_test.isnull().any(axis=1)]
prediction = clf_dt.predict(processed_test)
processed_test['Survived'] = prediction
processed_test.head()
processed_test.to_csv('out.csv', columns=['PassengerId', 'Survived'], index=False)