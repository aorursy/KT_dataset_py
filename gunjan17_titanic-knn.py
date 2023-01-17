# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn import linear_model

from sklearn.neighbors import KNeighborsClassifier

#for plotting the data

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train["Age"].isnull().sum()

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Age"].isnull().sum()
train["Embarked"].isnull().sum()

train["Embarked"]=train["Embarked"].fillna("S")


train.loc[train["Sex"] == "male", "Sex"] = 0

train.loc[train["Sex"]=="female","Sex"]=1



train.loc[train["Embarked"]=="S","Embarked"] =0

train.loc[train["Embarked"]=="C","Embarked"] =1

train.loc[train["Embarked"]=="Q","Embarked"] =2
new_col = ["Age","Sex","Embarked","Pclass","Fare","SibSp","Parch"]

x_train = train[new_col]

y_train = train["Survived"]
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
test["Age"].isnull().sum()

test["Age"] = test["Age"].fillna(train["Age"].median())

test["Age"].isnull().sum()
#cleanig the Embarked

test["Embarked"].isnull().sum()

test["Embarked"]=test["Embarked"].fillna("S")
test.loc[test["Sex"] == "male", "Sex"] = 0

test.loc[test["Sex"]=="female","Sex"]=1



test.loc[test["Embarked"]=="S","Embarked"] =0

test.loc[test["Embarked"]=="C","Embarked"] =1

test.loc[test["Embarked"]=="Q","Embarked"] =2
x_test = test[new_col]

#"Age","Sex","Embarked","Pclass","Fare"

test["Fare"] = test["Fare"].fillna(test["Fare"].median())

x_test = test[new_col]
a=knn.predict(x_test)

a = a.round()

l=[]

for i in a:

    i = int(i)

    l.append(i)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": a

    })

submission.to_csv('titanicknn.csv', index=False)