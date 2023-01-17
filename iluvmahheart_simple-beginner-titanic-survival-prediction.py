import pandas as pd 

import numpy as np
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
train.drop(['Name'],axis=1,inplace=True)
train.head()
train.columns
train.index
sample_sub=pd.read_csv("../input/gender_submission.csv")
sample_sub.head()
train.head()
test.head()
test.drop(['Name'],axis=1,inplace=True)
test.head()
train.isnull().sum()
train.index
test.isnull().sum()
train.drop(['Cabin'],axis=1,inplace=True)

test.drop(['Cabin'],axis=1,inplace=True)

test.drop(['Ticket'],axis=1,inplace=True)

train.drop(['Ticket'],axis=1,inplace=True)

train.drop(['PassengerId'],axis=1,inplace=True)

test.drop(['PassengerId'],axis=1,inplace=True)
test.head()
train.head()
import seaborn as sns

sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Parch',data=train)
sns.countplot(x='Survived',hue='SibSp',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
train.isnull().sum()
test.isnull().sum()
train['Age'].mean()
train['Age'].fillna((train['Age'].mean()), inplace=True)
test['Age'].fillna((test['Age'].mean()), inplace=True)
test['Fare'].fillna((test['Fare'].mean()), inplace=True)
train.dropna()
test.isnull().sum()
train.columns
train.head()
test.head()
Pclass=pd.get_dummies(train['Pclass'],drop_first=True)

Pclass1=pd.get_dummies(test['Pclass'],drop_first=True)

Sex=pd.get_dummies(train['Sex'],drop_first=True)

Sex1=pd.get_dummies(test['Sex'],drop_first=True)

Embarked=pd.get_dummies(train['Embarked'],drop_first=True)

Embarked1=pd.get_dummies(test['Embarked'],drop_first=True)
train=pd.concat([train,Pclass,Sex,Embarked],axis=1)

test=pd.concat([test,Pclass1,Sex1,Embarked1],axis=1)
train.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)

test.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)
train.head()
test.head()
sample_sub.head()
from sklearn.model_selection import train_test_split
y=train['Survived']
y.head()
X=train.drop('Survived',axis=1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
(145+67)/(41+67+15+145)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

y_pred = gbk.predict(X_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
test.head()
test.isnull().sum()
predictions1 = gbk.predict(test)

sample_sub['Survived']= predictions1

sample_sub.to_csv("submit.csv", index=False)

sample_sub.head()