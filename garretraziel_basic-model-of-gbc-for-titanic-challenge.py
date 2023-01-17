import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline



from sklearn import tree
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.info()
train_df.describe()
train_df.head()
test_df.head()
train_id = train_df[['PassengerId']]



train_id.head()
train_df = train_df.drop('PassengerId', axis=1)
train_df.isnull().sum()
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_id = test_df['PassengerId']



test_df = test_df.drop('PassengerId', axis=1)
test_df.isnull().sum()
train_df = train_df.drop('Cabin', axis=1)

test_df = test_df.drop('Cabin', axis=1)
test_df['Age'] = test_df['Age'].fillna(train_df['Age'].median())



test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df.mode(axis=0)['Embarked'].iloc[0])
train_df.head()
test_df.head()
train_df = train_df.drop('Ticket', axis=1)

test_df = test_df.drop('Ticket', axis=1)
X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']

X_test = test_df
from sklearn.preprocessing import StandardScaler, OneHotEncoder
X_train = X_train.drop('Name', axis=1)

X_test = X_test.drop('Name', axis=1)
X_train.loc[X_train['Sex'] == 'female', 'Sex'] = 0

X_train.loc[X_train['Sex'] == 'male', 'Sex'] = 1

X_test.loc[X_test['Sex'] == 'female', 'Sex'] = 0

X_test.loc[X_test['Sex'] == 'male', 'Sex'] = 1

X_train.loc[X_train['Embarked'] == 'S', 'Embarked'] = 1

X_train.loc[X_train['Embarked'] == 'C', 'Embarked'] = 2

X_train.loc[X_train['Embarked'] == 'Q', 'Embarked'] = 3

X_test.loc[X_test['Embarked'] == 'S', 'Embarked'] = 1

X_test.loc[X_test['Embarked'] == 'C', 'Embarked'] = 2

X_test.loc[X_test['Embarked'] == 'Q', 'Embarked'] = 3
X_train.head()
X_test.head()
X_train = pd.get_dummies(X_train, columns=['Pclass', 'Embarked'])



X_test = pd.get_dummies(X_test, columns=['Pclass', 'Embarked'])
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

cross_val_score(clf, X_train, Y_train, cv=10)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()

cross_val_score(clf, X_train, Y_train, cv=10)
from sklearn.svm import LinearSVC

clf = LinearSVC()

cross_val_score(clf, X_train, Y_train, cv=10)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

cross_val_score(clf, X_train, Y_train)
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()

cross_val_score(clf, X_train, Y_train, cv=10)
clf = GradientBoostingClassifier()

clf.fit(X_train, Y_train)

p = clf.predict(X_test)
p.shape
test_id.shape
result = pd.DataFrame(test_id)
result['Survived'] = p
result.to_csv('submission.csv', index=False)
clf = AdaBoostClassifier()

clf.fit(X_train, Y_train)

p = clf.predict(X_test)
result['Survived'] = p

result.to_csv('adaboostresult.csv', index=False)