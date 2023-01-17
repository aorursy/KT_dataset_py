import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train['Age']=train['Age'].median()



#Simple replace of missing Age values with the median of all the ages.
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.isna().sum()
test.isna().sum()
test['Age']=test['Age'].median()



#Perform the same of Age values of Test Data
train.drop(['Cabin'], axis=1, inplace=True)

test.drop(['Cabin'], axis=1, inplace=True)



#Drop the cabin values since there are too many values to impute
train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

test.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)



#Passenger ID and Ticket do not have an impact on predicting for survival, therefore removing those columns
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)



train.drop(['Sex','Embarked','Name'],axis=1,inplace=True)



train = pd.concat([train,sex,embark],axis=1)



#Embark and Sex columns can be turned to categorical columns using the get_dummies method
#Same to be done for Test data also



sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)



test.drop(['Sex','Embarked','Name'],axis=1,inplace=True)



test = pd.concat([test,sex,embark],axis=1)
train.head()
test.head()
test.Fare = test['Fare'].fillna(train['Fare'].median())
from sklearn.ensemble import RandomForestClassifier
# seperate the feature set and the target set

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)

X_test = test
from sklearn.metrics import accuracy_score



model = RandomForestClassifier(n_estimators=150)

model.fit(X_train, Y_train)

model.score(X_train, Y_train)
X_test=test

y_predict=model.predict(X_test)
random_forest = RandomForestClassifier(n_estimators=205)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest

test1=pd.read_csv('../input/titanic/test.csv')

submission = pd.DataFrame({

        "PassengerId": test1["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)
submission.count()