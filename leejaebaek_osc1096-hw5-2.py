# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid') 

%matplotlib inline 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")



train_df.head()
train_df.info()

print('-'*20)

test_df.info()
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test_df = test_df.drop(['Name','Ticket'], axis=1)
train_df['Pclass'].value_counts()
pclass_train_dummies = pd.get_dummies(train_df['Pclass'])

pclass_test_dummies = pd.get_dummies(test_df['Pclass'])



train_df.drop(['Pclass'], axis=1, inplace=True)

test_df.drop(['Pclass'], axis=1, inplace=True)



train_df = train_df.join(pclass_train_dummies)

test_df = test_df.join(pclass_test_dummies)
sex_train_dummies = pd.get_dummies(train_df['Sex'])

sex_test_dummies = pd.get_dummies(test_df['Sex'])



sex_train_dummies.columns = ['Female', 'Male']

sex_test_dummies.columns = ['Female', 'Male']



train_df.drop(['Sex'], axis=1, inplace=True)

test_df.drop(['Sex'], axis=1, inplace=True)



train_df = train_df.join(sex_train_dummies)

test_df = test_df.join(sex_test_dummies)
train_df["Age"].fillna(train_df["Age"].mean() , inplace=True)

test_df["Age"].fillna(train_df["Age"].mean() , inplace=True)
test_df["Fare"].fillna(0, inplace=True)
train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
train_df['Embarked'].value_counts()
test_df['Embarked'].value_counts()
train_df["Embarked"].fillna('S', inplace=True)

test_df["Embarked"].fillna('S', inplace=True)
embarked_train_dummies = pd.get_dummies(train_df['Embarked'])

embarked_test_dummies = pd.get_dummies(test_df['Embarked'])



embarked_train_dummies.columns = ['S', 'C', 'Q']

embarked_test_dummies.columns = ['S', 'C', 'Q']



train_df.drop(['Embarked'], axis=1, inplace=True)

test_df.drop(['Embarked'], axis=1, inplace=True)



train_df = train_df.join(embarked_train_dummies)

test_df = test_df.join(embarked_test_dummies)
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)