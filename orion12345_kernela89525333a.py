import pandas as pd

import numpy as np

X_train=pd.read_csv("../input/train.csv")

X_test=pd.read_csv("../input/test.csv")

X_train.columns
X_train.drop(['PassengerId'],axis=1,inplace=True)

X_test.drop(['PassengerId'],axis=1,inplace=True)

X_train.columns
X_train
X_train.drop(['Name'],axis=1,inplace=True)

X_test.drop(['Name'],axis=1,inplace=True)

X_train.drop(["Ticket"],axis=1,inplace=True)

X_test.drop(["Ticket"],axis=1,inplace=True)

X_train.drop(["Cabin"],axis=1,inplace=True)

X_test.drop(["Cabin"],axis=1,inplace=True)

print(X_train)
print(X_train.columns)
X_train.head()
X_train.isnull().sum()
X_test.isnull().sum()
X_test['Age'].fillna((X_test['Age'].mean()),inplace=True)
X_train['Age'].fillna((X_train['Age'].mean()),inplace=True)
X_test['Fare'].fillna((X_test['Fare'].mean()),inplace=True)
X_train.head()
a1=pd.get_dummies(X_train["Pclass"],drop_first=True)

a2=pd.get_dummies(X_test["Pclass"],drop_first=True)

a3=pd.get_dummies(X_train["Sex"],drop_first=True)

a4=pd.get_dummies(X_test["Sex"],drop_first=True)

a5=pd.get_dummies(X_train["Embarked"],drop_first=True)

a6=pd.get_dummies(X_test["Embarked"],drop_first=True)
X_train=pd.concat([X_train,a1,a3,a5],axis=1)

X_test=pd.concat([X_test,a2,a4,a6],axis=1)

X_train.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)

X_test.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)
X_train.columns
from sklearn.model_selection import train_test_split

X_train.columns

y=X_train['Survived']
X_train.drop(['Survived'],axis=1,inplace=True)
X=X_train
X_train, X_t, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_t)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
(145+67)/(41+67+15+145)
sample_sub=pd.read_csv("../input/gender_submission.csv")

predictions1 = logmodel.predict(X_test)

#print(predictions1.shape)

sample_sub['Survived']

sample_sub.to_csv("submit.csv", index=False)

sample_sub.head()