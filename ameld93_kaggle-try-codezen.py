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
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")





train.head()
sns.heatmap(train.isnull(), )
sns.countplot(x = "Survived", data= train) 
sns.countplot(x = "Survived", hue="Sex", data= train)
sns.countplot(x= "Survived", hue="Sex", data=train[train["Age"] < 16])
sns.distplot(train["Fare"])
sns.countplot(x="Survived", hue="Pclass", data=train)
sns.distplot(train["Age"].dropna(), bins = 50)
sns.boxplot(x="Pclass", y="Age", data=train)
train[train["Pclass"] == 1]["Age"].median()
train[train["Pclass"] == 2]["Age"].median()
train[train["Pclass"] == 3]["Age"].median()
def replace_age(row):

    Age = row[0]

    Pclass = row[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        if Pclass == 2:

            return 29

        if Pclass == 3:

            return 24

    else:

        return Age

        
train["Age"] = train[["Age", "Pclass"]].apply(replace_age, axis = 1)
train = train.drop("Cabin", axis=1)
train.drop(["Name", "Ticket"], axis=1, inplace=True)
from sklearn.ensemble import RandomForestClassifier
train["Embarked"] = train["Embarked"].dropna()
test["Age"] = test[["Age", "Pclass"]].apply(replace_age, axis = 1)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

test = test.drop("Cabin", axis=1)

test.drop(["Name", "Ticket"], axis=1, inplace=True)
test.head()
train.head()
train.head(2)
emark = pd.get_dummies(train["Embarked"], drop_first=True)

sex = pd.get_dummies(train["Sex"], drop_first=True)

train = pd.concat([train, emark, sex], axis = 1)

train.drop(["Sex", "Embarked"], axis = 1, inplace=True)



emark = pd.get_dummies(test["Embarked"], drop_first=True)

sex = pd.get_dummies(test["Sex"], drop_first=True)

test = pd.concat([test, emark, sex], axis = 1)

test.drop(["Sex", "Embarked"], axis = 1, inplace=True)

x_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
rf = RandomForestClassifier()

rf.fit(x_train, y_train)
prediction = rf.predict(test)
output = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
output.to_csv("titanic.csv", index=False)