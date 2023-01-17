# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head(5)
train.shape
train.columns
train.info()
train.describe()
train.isnull().sum()
test.isnull().sum()
# train["Age"]=train["Age"].fillna(train["Age"].mean())
train["Age"].fillna(train["Age"].mean() , inplace=True)
train["Age"].mean()
train.isnull().sum()
drop = ["PassengerId" , "ticket drop" , "cabin", "name"]
analyse =["Pclass" , "gender" , "age" , "sibsp" , "parch"  , "fare" , "embarked"]
def bargraph(feature):
    survived = train[train["Survived"]==1][feature].value_counts()
    dead = train[train["Survived"]==0][feature].value_counts()
    
    df = pd.DataFrame([survived , dead])
    df.plot(kind="bar" , stacked=True)
bargraph("Sex")

bargraph("Pclass")
bargraph("SibSp")
bargraph("Parch")
bargraph("Embarked")
train[train["Survived"]==0]["Sex"].value_counts()
train["Embarked"].value_counts()
train["Embarked"].value_counts
train.loc[train["Sex"]=="male","Sex"]=1
train.loc[train["Sex"]=="female","Sex"]=0

train.loc[train["Embarked"]=="S","Embarked"]=0
train.loc[train["Embarked"]=="C","Embarked"]=1
train.loc[train["Embarked"]=="Q","Embarked"]=2

test.loc[test["Sex"]=="male","Sex"]=1
test.loc[test["Sex"]=="female","Sex"]=0

test.loc[test["Embarked"]=="S","Embarked"]=0
test.loc[test["Embarked"]=="C","Embarked"]=1
test.loc[test["Embarked"]=="Q","Embarked"]=2

# 16    0
# 16 26 1
# 26 36 2
# 36 62 3

train.loc[train["Age"]<=16,"Age"]=0
train.loc[(train["Age"]>16) & (train["Age"]<=26),"Age"] = 1
train.loc[(train["Age"]>26) & (train["Age"]<=36),"Age"] = 2
train.loc[(train["Age"]>36) & (train["Age"]<=66),"Age"] = 3
train.loc[(train["Age"]>66),"Age"] = 4


test.loc[test["Age"]<=16,"Age"]=0
test.loc[(test["Age"]>16) & (test["Age"]<=26),"Age"] = 1
test.loc[(test["Age"]>26) & (test["Age"]<=36),"Age"] = 2
test.loc[(test["Age"]>36) & (test["Age"]<=66),"Age"] = 3
test.loc[(test["Age"]>66),"Age"] = 4
train.drop(["PassengerId" , "Name" , "Ticket" , "Cabin"] , inplace=True , axis=1)
test.drop(["PassengerId" , "Name" , "Ticket" , "Cabin"] , inplace=True , axis=1)

train.info()
train.dropna(axis=0,inplace=True)
test.dropna(axis=0,inplace=True)


train.info()
test["Age"].value_counts()
bargraph("Age")
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']],train["Survived"])


pred = clf.predict(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']])

(pred == train["Survived"]).sum()/train.shape[0]
train.shape[0]
