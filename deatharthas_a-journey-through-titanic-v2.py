import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
train_pd = pd.read_csv("../input/train.csv")
train_pd.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)

train_pd.drop(['SibSp','Parch','Fare','Cabin','Embarked'],axis=1,inplace=True)
age_median = np.median(train_pd['Age'][train_pd['Age']>0])

train_pd['Age']=train_pd['Age'].fillna(age_median)
sex_dummies = pd.get_dummies(train_pd['Sex'])

sex_dummies.columns = ['Femal','Male']

train_pd.drop(['Sex'],axis=1,inplace=True)

train_pd = train_pd.join(sex_dummies)
X_train = train_pd.drop(['Survived'],axis=1)

Y_train = train_pd['Survived']

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

knn.score(X_train, Y_train)
test_pd = pd.read_csv("../input/test.csv")

X_test = test_pd.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
X_test['Age'] = X_test['Age'].fillna(np.median(X_test['Age'][X_test['Age']>0]))
test_dummies = pd.get_dummies(X_test['Sex'])

test_dummies.columns=['Female','Male']

X_test = X_test.join(test_dummies)

X_test.drop(['Sex'],axis=1,inplace=True)
Y_pred = knn.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test_pd["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)