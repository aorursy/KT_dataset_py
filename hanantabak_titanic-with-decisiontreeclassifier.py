# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train.head()
train.info()
unnecessary = ['Name','Ticket','Cabin']

train = train.drop(unnecessary, axis=1)

test = test.drop(unnecessary, axis=1)

test
train['male']= pd.get_dummies(train['Sex'], drop_first=True)

train = train.drop('Sex',axis=1)

train['Embarked']= pd.get_dummies(train['Embarked'])

test['male']= pd.get_dummies(test['Sex'], drop_first=True)

test = test.drop('Sex',axis=1)

test['Embarked']= pd.get_dummies(test['Embarked'])

train.info()

test.info()
train['Age'].fillna(train['Age'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)

test['Fare'].fillna(test['Fare'].median(), inplace=True)



train.info()

test.info()
train.corr()
features = ['male','Fare','Pclass','Embarked','Parch']

Xtrain = train[features]

Xtest = test[features]

y = train['Survived']
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

model = DecisionTreeClassifier(max_depth=3)

model.fit(Xtrain,y)

print(model.score(Xtrain,y))

pred = model.predict(Xtest)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':pred})

output.to_csv('DecisionTreeClasifier4.csv',index=False)