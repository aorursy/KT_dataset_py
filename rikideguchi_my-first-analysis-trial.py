import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mpl

%matplotlib inline

import warnings 

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head()
train_df.describe()
train_df.info()
train_df['Sex'][train_df['Sex'] == 'male'] = 0

train_df['Sex'][train_df['Sex'] == 'female'] = 1

test_df['Sex'][test_df['Sex'] == 'male'] = 0

test_df['Sex'][test_df['Sex'] == 'female'] = 1
list(train_df.columns)
train_df = train_df.drop(columns = ['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.drop(columns = ['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df
random_forest = RandomForestClassifier(n_estimators = 100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)