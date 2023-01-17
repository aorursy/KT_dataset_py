import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

tdata=pd.read_csv('../input/train.csv')

x_test=pd.read_csv('../input/test.csv')

y_test=pd.read_csv('../input/gender_submission.csv')

train=pd.DataFrame(tdata)

train.info()
train.head()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

le.fit(train['Sex'])

Sex=le.transform(train['Sex'])

sex=pd.DataFrame(Sex,)

train=pd.concat([train,sex],axis=1)

train.columns=['PassengerId','Survived',  'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','sex']

train.columns
le=LabelEncoder()

le.fit(x_test['Sex'])

Sex=le.transform(x_test['Sex'])

sex=pd.DataFrame(Sex)

X_test=pd.concat([x_test,sex],axis=1)

X_test.columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','sex']

X_test=X_test.fillna(0)

xtest=X_test[['Pclass','Age','SibSp','Parch','Fare','sex']]
data=train[['Survived','Pclass','Age','SibSp','Parch','Fare','sex']]

xtr=train[['Pclass','Age','SibSp','Parch','Fare','sex']]

ytr=train['Survived']

Ytr=pd.DataFrame(ytr)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix,mean_squared_error,mean_absolute_error

model=DecisionTreeClassifier()

model.fit(xtr,Ytr)

predict=model.predict(xtest)

predict
Ytest=y_test['Survived']

Ytest.head()
print("accuracy_score:", accuracy_score(Ytest,predict))
confusion_matrix(Ytest,predict)
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()

model.fit(xtr,Ytr)

predict=model.predict(xtest)

accuracy_score(Ytest,predict)

print("accuracy_score:", accuracy_score(Ytest,predict))
from sklearn.ensemble import BaggingClassifier

model=BaggingClassifier()

model.fit(xtr,Ytr)

predict=model.predict(xtest)

accuracy_score(Ytest,predict)

print("accuracy_score:", accuracy_score(Ytest,predict))
from sklearn.ensemble import GradientBoostingClassifier

model=GradientBoostingClassifier()

model.fit(xtr,Ytr)

predict=model.predict(xtest)

accuracy_score(Ytest,predict)

print("accuracy_score:", accuracy_score(Ytest,predict))
from sklearn.ensemble import AdaBoostClassifier

model=AdaBoostClassifier()

model.fit(xtr,Ytr)

predict1=model.predict(xtest)

accuracy_score(Ytest,predict)

print("accuracy_score:", accuracy_score(Ytest,predict))
confusion_matrix(Ytest,predict1)
submission = pd.DataFrame({

        "PassengerId": x_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)
filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)