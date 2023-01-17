# -*- coding: utf-8 -*-

"""

Created on Tue Jun 13 12:05:20 2017



@author: rajes

"""

import os

import pandas as pd

import numpy as np



#os.chdir('C:/Users/rajes/OneDrive/RAJESH/Projects/titanic')

traindataset = pd.read_csv('../input/train.csv')

testdataset = pd.read_csv('../input/test.csv')



#traindataset.describe()



#traindataset.info()



#traindataset.head()

X_train = traindataset.drop(['Survived', 'Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)

#X_train = X_train.iloc[:,:].values

X_train["Age"] = X_train["Age"].fillna(X_train["Age"].mean())

X_train["Embarked"] = X_train["Embarked"].fillna('S')

X_test = testdataset.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)

X_test["Age"] = X_test["Age"].fillna(X_test["Age"].mean())

X_test["Fare"] = X_test["Fare"].fillna(X_test["Fare"].mean())

#X_test = X_test.iloc[:,:].values



Y_train = traindataset["Survived"]



#pd.isnull(X_train).sum() > 0

#pd.isnull(X_test).sum() > 0

#

#X_train['Embarked'].value_counts().idxmax()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X_train.iloc[:, 1] = labelencoder_X.fit_transform(X_train.iloc[:, 1])

labelencoder_test = LabelEncoder()

X_test.iloc[:, 1] = labelencoder_test.fit_transform(X_test.iloc[:, 1])

#

#onehotencoder = OneHotEncoder(categorical_features = [1])

#X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X1 = LabelEncoder()

X_train.iloc[:, -1] = labelencoder_X1.fit_transform(X_train.iloc[:, -1])

labelencoder_test1 = LabelEncoder()

X_test.iloc[:, -1] = labelencoder_test1.fit_transform(X_test.iloc[:, -1])



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = random_forest.score(X_train, Y_train) * 100

acc_random_forest



submission = pd.DataFrame({

        "PassengerId": testdataset["PassengerId"],

        "Survived": Y_pred

    })

#submission.to_csv('../output/submission.csv', index=False)