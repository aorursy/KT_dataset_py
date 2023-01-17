# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler





%matplotlib inline
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.shape
test.shape
train.info()
train['Cabin'].value_counts()
train['Age'].fillna(train['Age'].mean(), inplace=True)

train.drop('Cabin', axis =1, inplace=True)

train.drop(train[train.isna()['Embarked']].index, inplace=True, axis =0)
train.isna().sum()
test['Age'].fillna(train['Age'].mean(), inplace=True)

test.drop('Cabin', axis =1, inplace=True)
test.isna().sum()
test[test['Fare'].isna()].index
test.iloc[152,8] = np.round(train['Fare'].mean(),2)
train.drop('Name', axis =1, inplace=True)

test.drop('Name', axis =1, inplace=True)
train.drop('Ticket', axis=1, inplace=True)

test.drop('Ticket', axis=1, inplace=True)
train.head()
train['Sex'].value_counts()
train['Embarked'].value_counts()
train['Sex'] = train['Sex'].apply(lambda x: '1' if x == 'male' else '0')

train['Embarked'] = train['Embarked'].apply(lambda x: '0' if x == 'S' else ('1' if x == 'C' else '2'))
train['Sex'] = train['Sex'].astype('int64')

train['Embarked'] = train['Embarked'].astype('int64')

train['Age'] = train['Age'].astype('int64')
train.set_index('PassengerId', inplace=True)
test['Sex'] = test['Sex'].apply(lambda x: '1' if x == 'male' else '0').astype('int64')

test['Embarked'] = test['Embarked'].apply(lambda x: '0' if x == 'S' else ('1' if x == 'C' else '2')).astype('int64')

test['Age'] = test['Age'].astype('int64')

test.set_index('PassengerId', inplace=True)
gender_submission.set_index('PassengerId', inplace=True)
X_train = train[train.columns[train.columns != 'Survived']]

X_test = test[:]

y_train = train['Survived']

y_test = gender_submission['Survived']
col = X_train.columns[X_train.columns != 'Fare']

for i in col:

    print('*'* 30)

    print(i)

    print(X_train[i].value_counts())
ohe = OneHotEncoder(handle_unknown='ignore')

X_train_ohe = ohe.fit_transform(X_train[col])

X_test_ohe = ohe.transform(X_test[col])
scr = MinMaxScaler()

X_train_mm = scr.fit_transform(X_train['Fare'].values.reshape(-1,1))

X_test_mm = scr.transform(X_test['Fare'].values.reshape(-1,1))
from scipy.sparse import hstack
X_train_fin = hstack((X_train_ohe, X_train_mm))

X_test_fin = hstack((X_test_ohe, X_test_mm))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

from sklearn.model_selection import GridSearchCV
C =  [ 10**-5, 10**-4, 10**-3, 10**-2, 0.1, 1, 10, 100 ]

lr = LogisticRegression(class_weight= 'balanced', n_jobs=-1)

clf = GridSearchCV(lr, param_grid= {'C':C}, n_jobs=-1, cv=10, scoring='accuracy')

clf.fit(X_train_fin, y_train.values)
alpha = clf.best_params_
alpha.values()
clf = LogisticRegression(C= 1, class_weight='balanced', n_jobs=-1)

clf.fit(X_train_fin, y_train)
y_pred = clf.predict(X_test_fin)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
confusion_matrix(y_test, y_pred)