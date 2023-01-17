# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.info()

test.info()

train.describe()

train.drop(['PassengerId', 'Ticket'],axis = 1, inplace = True)

test.drop(['Ticket'],axis = 1,inplace = True)

#Name 

train['Title'] = train["Name"].map(lambda name : name.split(".")[0].split(" ")[-1])

test['Title'] = test["Name"].map(lambda name : name.split(".")[0].split(" ")[-1])

train["Title"] = train["Title"].map({"Mr" : "Mr", "Mrs" : "Mrs", "Miss" : "Miss", "Master" : "Master"})

train["Title"].fillna("Others", inplace=True)

test["Title"] = test["Title"].map({"Mr" : "Mr", "Mrs" : "Mrs", "Miss" : "Miss", "Master" : "Master"})

test["Title"].fillna("Others", inplace=True)

train["Title"] = train["Title"].map({"Mr" : 0, "Mrs" : 1, "Miss" : 2, "Master" : 3, "Others" : 4})

test["Title"] = test["Title"].map({"Mr" : 0, "Mrs" : 1, "Miss" : 2, "Master" : 3, "Others" : 4})

train.drop(['Name'],axis = 1, inplace = True)

test.drop(['Name'],axis = 1, inplace = True)

#Sex

print("Total Null Entries in training samples :", train['Sex'].isnull().sum())

print("Total Null Entries in testing samples  :", test['Sex'].isnull().sum())

train['Sex'] = train['Sex'].map({"male" : 0,"female" : 1})

test['Sex'] = test['Sex'].map({"male" : 0,"female" : 1})

#Age

train["AgeCategory"] = "Adult"

train["AgeCategory"].loc[train["Age"] < 18 ] = "Child"

train["AgeCategory"].loc[train["Age"] > 50 ] = "Old"

train["AgeCategory"].loc[train["Age"].isnull()] = "MissingData"

train["AgeCategory"] = train["AgeCategory"].map({"Adult": 0,"Child" : 1,"Old": 2,"MissingData": 3})

test["AgeCategory"] = "Adult"

test["AgeCategory"].loc[train["Age"] < 18 ] = "Child"

test["AgeCategory"].loc[train["Age"] > 50 ] = "Old"

test["AgeCategory"].loc[train["Age"].isnull()] = "MissingData"

test["AgeCategory"] = test["AgeCategory"].map({"Adult": 0,"Child" : 1,"Old": 2,"MissingData": 3})

train.drop(['Age'],axis = 1, inplace = True)

test.drop(['Age'],axis = 1, inplace = True)

#Family

train['Family'] = train['SibSp'] + train['Parch'] + 1

train['FamilySize'] = train['Family']

train['FamilySize'].loc[train['Family'] == 1] = "Small"

train['FamilySize'].loc[train['Family'] > 1] = "Medium"

train['FamilySize'].loc[train['Family'] > 5] = "Large"

train["FamilySize"] = train["FamilySize"].map({"Small": 0,"Medium" : 1,"Large": 2})

test['Family'] = test['SibSp'] + test['Parch'] + 1

test['FamilySize'] = test['Family']

test['FamilySize'].loc[test['Family'] == 1] = "Small"

test['FamilySize'].loc[test['Family'] > 1] = "Medium"

test['FamilySize'].loc[test['Family'] > 5] = "Large"

test["FamilySize"] = test["FamilySize"].map({"Small": 0,"Medium" : 1,"Large": 2})



train.drop(['SibSp','Parch','Family'],axis = 1, inplace = True)

test.drop(['SibSp','Parch','Family'],axis = 1, inplace = True)

#Fare

test['Fare'].fillna(train['Fare'].mean(), inplace=True)

scale = preprocessing.MinMaxScaler()

train['normalizedFare'] = scale.fit_transform(train['Fare'].reshape(-1,1))

test["normalizedFare"] = scale.transform(test['Fare'].reshape(-1,1))

train.drop("Fare", axis=1, inplace=True)

test.drop("Fare", axis=1, inplace=True)

#Embarked

train['Embarked'].fillna('S', inplace=True)

test['Embarked'].fillna('S', inplace=True)

train['Embarked'] = train['Embarked'].map({"S" : 0,"C": 1, "Q": 3})

test['Embarked'] = test['Embarked'].map({"S" : 0,"C": 1, "Q": 3})

#cabin

train.drop(["Cabin"],axis = 1, inplace = True)

test.drop(["Cabin"],axis = 1, inplace = True)

#Splitting for evaluation

X_train = train.drop(["Survived"],axis = 1)

y_train = train["Survived"]

X_test = test.drop(["PassengerId"],axis = 1).copy()

# Any results you write to the current directory are saved as output.

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf',random_state = 0)

classifier.fit(X_train,y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)