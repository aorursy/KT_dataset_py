# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train.drop(columns=cols, axis=1, inplace=True)

cols=['Name', 'Ticket', 'Cabin']
test.drop(columns=cols, axis=1, inplace=True)
test.head()

train.head()
train['Age'].describe()
train.head()
train['Age'] = train['Age'].fillna(0.0)
test['Age'] = test['Age'].fillna(0.0)
train['Age'] = train['Age'].round(2)
test['Age'] = test['Age'].round(2)
median = train['Age'].median()
train['Age'] = train['Age'].replace(0.0, median)
median = test['Age'].median()
test['Age'] = test['Age'].replace(0, median)
test.describe()
train['Age'] = train['Age'].astype('int64')
test['Age'] = test['Age'].astype('int64')
def round_(df, variable):
    df[variable] = df[variable].round()
    df[variable] = df[variable].astype('int64')
round_(train, 'Fare')
round_(test, 'Fare')

print(train['Fare'])
print(test['Fare'])
train['Fare'].describe()
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
print(train.info())
print('_'*40)
print(test.info())
test['Fare'] = test['Fare'].astype('int64')
test.info()
train.head()
sex = train['Sex'].unique()
for i in range(len(sex)):
    train.loc[train['Sex'] == sex[i], 'Sex'] = i
train
sex = test['Sex'].unique()
for i in range(len(sex)):
    test.loc[test['Sex'] == sex[i], 'Sex'] = i
test
unique_emb = train['Embarked'].unique()
for i in range(len(unique_emb)):
    train.loc[train['Embarked'] == unique_emb[i], 'Embarked'] = i
train.Embarked
unique_emb = test['Embarked'].unique()
for i in range(len(unique_emb)):
    test.loc[test['Embarked'] == unique_emb[i], 'Embarked'] = i
test.Embarked
def ty(df, variable):
    df[variable] = df[variable].astype('int64')
    
ty(train, 'Sex')
ty(train, 'Embarked')
ty(test, 'Sex')
ty(test, 'Embarked')
print(train.info())
print('_'*50)
print(test.info())
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('Titanic.csv', index=False)