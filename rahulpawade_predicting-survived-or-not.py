import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
train = pd.read_csv('../input/titanic/train.csv')
train.shape
train.info()
train.describe()
train.head()
train['Age'].plot.box()
sns.distplot(train['Age'])
train['Age'].fillna(train['Age'].mean(),inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
train.loc[train['Age']>=18,'Adult']=1

train.loc[train['Age']<18,'Adult']=0
train.loc[train['Fare']>=31,'Rich']=1

train.loc[train['Fare']<31,'Rich']=0
train
sns.distplot(train['Fare'])
train['Fare'].describe()
train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Age','Fare'],axis=1,inplace=True)
train = pd.get_dummies(train,drop_first=True)
train.count()
train = train.astype(np.int)
sns.heatmap(train.corr(),annot=True)
test = pd.read_csv('../input/titanic/test.csv')
test.info()
test['Age'].fillna(test['Age'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test.loc[test['Age']>=18,'Adult']=1

test.loc[test['Age']<18,'Adult']=0

test.loc[test['Fare']>=31,'Rich']=1

test.loc[test['Fare']<31,'Rich']=0
test
test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Age','Fare'],axis=1,inplace=True)
test = pd.get_dummies(test,drop_first=True)
test = test.astype(np.int)
test.count()
x_train = train.drop('Survived',axis=1)

y_train = train['Survived']

x_test = test
from sklearn.linear_model import LogisticRegression

model =  LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
s = pd.read_csv("../input/titanic/gender_submission.csv")
f = {"PassengerId":s["PassengerId"],"Survived":y_pred}

f = pd.DataFrame(f)
f.head()
f.to_csv('output.csv',index=False)