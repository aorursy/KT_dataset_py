import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train=pd.read_csv('../input/titanic/train.csv')

train.head()
test=pd.read_csv('../input/titanic/test.csv')

test.head()
train=pd.get_dummies(train,columns=['Sex','Pclass','Embarked'])

train.head()
plt.figure(figsize=(20,10))

sns.heatmap(train.corr(), annot=True, vmin=0, vmax=1, center= 0)
train.isnull().sum()
train['Age'].fillna(train['Age'].mean(),inplace=True)

train.isnull().sum()
X_train=train.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1)

X_train.head()
y_train=train.Survived

y_train.head()
test=pd.get_dummies(test,columns=['Sex','Pclass','Embarked'])

test.head()
X_test=test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

X_test.isnull().sum()
X_test.dtypes
X_test['Age'].mean()
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)

X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)

X_test.isnull().sum()
X_test.dtypes
X_test.isnull().sum()
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(X_train,y_train)

log.predict(X_test)

log.score(X_train,y_train)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

dt.predict(X_test)

dt.score(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

rf.predict(X_test)

rf.score(X_train,y_train)
y_test = rf.predict(X_test)

y_test.shape
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_test})

output.to_csv('submission.csv', index=False)