import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

sns.set()

%matplotlib inline
train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")
train.head()
train.describe()
train.shape
train.info()
test.head()
test.describe()
test.shape
test.info()
'''Delete the Column name Cabin because it has little information as of the values ar NaN'''

train.drop(['Cabin'], axis=1, inplace=True)

test.drop(['Cabin'], axis=1, inplace=True)
sns.countplot(x='Survived', data=train)
sns.countplot(x='Pclass', data=train)
sns.countplot(x='Sex', data=train)
plt.scatter(range(train.shape[0]),np.sort(train['Age']))
sns.countplot(x='SibSp', data=train)
sns.countplot(x='Parch', data=train)
sns.countplot(x='Embarked', data=train)
'''Dealing with the missing values in feature Age'''

train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())
print(train['Sex'].unique())
'''Converting the datatype to int, so it can be used to fit the classifier'''

train.loc[train['Sex']=='male', 'Sex'] = 0

train.loc[train['Sex']=='female', 'Sex'] = 1



test.loc[test['Sex']=='male', 'Sex'] = 0

test.loc[test['Sex']=='female', 'Sex'] = 1
train.loc[train['Embarked']=='S', 'Embarked'] = 0

train.loc[train['Embarked']=='C', 'Embarked'] = 1

train.loc[train['Embarked']=='Q', 'Embarked'] = 2

train['Embarked'] = train['Embarked'].fillna(0)





test.loc[test['Embarked']=='S', 'Embarked'] = 0

test.loc[test['Embarked']=='C', 'Embarked'] = 1

test.loc[test['Embarked']=='Q', 'Embarked'] = 2

test['Embarked'] = test['Embarked'].fillna(0)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
train.head()
train.info()
test.head()
test.info()
'''getting features and labels'''

train_X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

train_y = train[['Survived']]



test_X = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
X,x,Y,y = train_test_split(train_X, train_y, test_size=0.2)



clf = LogisticRegression()

clf.fit(X,Y)

pred = clf.predict(x)
accuracy = accuracy_score(pred, y)

print(accuracy)
clf.fit(train_X, train_y)

pred_values = clf.predict(test_X)

pred_values
sns.countplot(pred_values)