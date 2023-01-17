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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
ID = test['PassengerId']
train.head(3)
test.head(3)
train.info()
test.info()
# fill nan with median - right skewed distrubution
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
sns.distplot(train['Age'])
sns.distplot(test['Age'])
sns.distplot(test['Fare'])
# 2 rows with NAN values are droped
train['Embarked'].dropna(inplace=True)
# Cabin Contains only 22% data 
204/891
# Cabin Contains only 22% data
91/418
train.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
test.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
train.info()
# with family = 1
# without = 0
train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']
train['Family'].unique()
test['Family'].unique()
train['Family'] = train['Family'].apply(lambda x: '1' if x > 0 else '0')
test['Family'] = test['Family'].apply(lambda x: '1' if x > 0 else '0')
train['Family'] = train['Family'].astype(int)
test['Family'] = test['Family'].astype(int)
train.drop(['SibSp', 'Parch'], axis='columns', inplace=True)
test.drop(['SibSp', 'Parch'], axis='columns', inplace=True)
train.info()
test.info()
train = pd.get_dummies(data=train, columns=['Embarked', 'Family', 'Sex', 'Pclass'], drop_first=True)
test = pd.get_dummies(data=test, columns=['Embarked', 'Family', 'Sex', 'Pclass'], drop_first=True)
test.head()
train.head()
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
X_test = test.values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV
parameter = [
    {'C': [1, 20, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 20, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    {'C': [1, 20, 100, 1000], 'kernel': ['ploy'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    {'C': [1, 20, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameter, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
accuracy = grid_search.best_score_
parameter = grid_search.best_params_
accuracy
parameter
parameter = [
    {'C': [1, 2, 3, 4, 5, 6], 'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]},
]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameter, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']
y_pred = pd.merge(ID, y_pred, left_index=True, right_index=True)
y_pred
