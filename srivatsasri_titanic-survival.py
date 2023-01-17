# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.shape
test_data.shape

test_data.head()
train_data.Pclass.value_counts().plot(kind='bar')
train_data.isnull().any()
train = train_data.drop(['Name','Ticket','Cabin','Fare','PassengerId'],axis=1)
train.head(3)
train.Sex = train.Sex.map({'female':0,'male':1})
train.loc[train.Age.isnull(),'Age'] = train.Age.median()
train.loc[train.Embarked.isnull(),'Embarked'] = train.Embarked.mode()
train = pd.get_dummies(train, columns=['Pclass','Embarked'])
train.head(1)
train.isnull().any()
y = train['Survived']
y.__len__()
X = train.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

kfold = KFold(n_splits=10, random_state=0)
dTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dTree.fit(X_train,y_train)
scoring = 'accuracy'
results = cross_val_score(dTree,X_train,y_train, cv=kfold, scoring=scoring)
acc_dt = results.mean()
dt_std = results.std()
acc_dt
y_pred = dTree.predict(X_test)
accuracy_score(y_test,y_pred)
test_data.isnull().any()
test_data.loc[test_data.Age.isnull(),'Age'] = test_data.Age.median()
test = test_data.drop(['Name','Ticket','Cabin','Fare','PassengerId'],axis=1)
test.Sex = test.Sex.map({'female':0,'male':1})
test = pd.get_dummies(test, columns=['Pclass','Embarked'])
test.head(2)
prediction = dTree.predict(test)
prediction.__len__()
submission = pd.DataFrame({"PassengerId":test_data["PassengerId"],"Survived":prediction})
submission.to_csv("sample_submission.csv",index=False)  
