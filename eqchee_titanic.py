# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Data analysis and handling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation

import matplotlib.pyplot as plt

%matplotlib inline



# Machine Learning

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
train.shape
train.describe(include='all')
train['Pclass'].value_counts()
train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()
pd.crosstab(train['Pclass'], train['Survived'])
train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()
bins = [0, 20, 40, 60, 80]

train[['Age','Survived']].groupby(['Survived', pd.cut(train.Age, bins)]).size().unstack()
bins_fare = [0, 8, 15, 30, 513]

train[['Fare','Survived']].groupby(['Survived', pd.cut(train.Fare, bins_fare)]).size().unstack()
train[['Fare','Pclass']].groupby(['Pclass', pd.cut(train.Fare, bins_fare)]).size().unstack()
train['Embarked'].value_counts()
#data[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()

pd.crosstab(train['Embarked'], train['Survived'])
test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)

test.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

data = [train, test]
for dataset in data:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(test['Title'], test['Sex'])
for dataset in data:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Uncommon')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

pd.crosstab(train['Title'], train['Sex'])
train = train.drop('Name', axis=1)

test = test.drop('Name', axis=1)

train.head()
train[train['Age'].isnull()]
mean_age = train.groupby(['Title'])['Age'].mean()

mean_age
train['Age'].fillna(train.groupby(['Title']).transform('mean').Age, inplace=True)

test['Age'].fillna(test.groupby(['Title']).transform('mean').Age, inplace=True)
train.Age.isnull().sum()
test.Age.isnull().sum()
train = train.fillna(train['Embarked'].value_counts().index[0])

train.Embarked.isnull().sum()
test['Fare'].fillna(test.groupby(['Pclass']).transform('mean').Fare, inplace=True)
train.dtypes
test.dtypes
train['Pclass'] = train['Pclass'].astype(str)

train.dtypes
test['Pclass'] = test['Pclass'].astype(str)

test.dtypes
train = pd.get_dummies(train)

train.head()
test = pd.get_dummies(test)

test.head()
X = train.drop(['Survived'], axis=1)

y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

mms = MinMaxScaler()

X_train_scaled = mms.fit_transform(X_train)

X_val_scaled = mms.transform(X_val)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train_scaled,y_train)

knn.score(X_val_scaled, y_val)
logreg = LogisticRegression()

logreg.fit(X_train_scaled, y_train)

logreg.score(X_val_scaled, y_val)
rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

rf.score(X_val, y_val)
svm = SVC(C=100)

svm.fit(X_train_scaled, y_train)

svm.score(X_val_scaled, y_val)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],

              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(SVC(), param_grid, cv=5)

grid.fit(X_train_scaled, y_train)

grid.best_params_
grid.best_score_
param_grid_rf = {'max_features': [1, 2, 3]}

grid_rf = GridSearchCV(RandomForestClassifier(n_estimators=100), param_grid_rf, cv=5)

grid_rf.fit(X_train, y_train)

grid_rf.best_params_
grid_rf.best_score_
svc = SVC(C=1, gamma=1)

svc.fit(X_train_scaled,y_train)

X_test = test.drop(['PassengerId'], axis=1)

X_test_scaled = mms.transform(X_test)

y_test = svc.predict(X_test_scaled)
submission = pd.DataFrame({

    'PassengerId': test['PassengerId'],

    'Survived': y_test })

submission.to_csv('submission.csv', index=False)