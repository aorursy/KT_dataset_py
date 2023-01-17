# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
df.head()
df.info()
df.isnull().sum()
df.columns.tolist()
df["TravelAlone"] = df["SibSp"] + df["Parch"]
test["TravelAlone"] = test["SibSp"] + test["Parch"]
list = "PassengerId","Name","SibSp","Parch","Ticket","Cabin","Fare"
df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Fare"],axis=1,inplace=True)
test.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Fare"],axis=1,inplace=True)
df.head()
df['Sex'].replace(['male','female'],[0,1],inplace=True)

df['Embarked'].replace(['S','C',"Q"],[0,1,2],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)

test['Embarked'].replace(['S','C',"Q"],[0,1,2],inplace=True)
print(df.head())

print("*"*30)

print(test.head())
df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean()
df[["Sex","Survived"]].groupby(["Sex"],as_index=0).mean()
df[["TravelAlone","Survived"]].groupby(["TravelAlone"],as_index=0).mean().sort_values(by="Survived",ascending=False)
g = sns.FacetGrid(df,col="Survived")

g.map(plt.hist, "Age",bins=20)
g = sns.FacetGrid(df,col="Survived",row="Pclass",size=2.4,aspect=1.6)

g.map(plt.hist, "Age",bins=20)
g = sns.FacetGrid(df,row="Embarked",size=2.2,aspect=1.6)

g.map(sns.pointplot,"Pclass","Survived","Sex")

g.add_legend()
print(df.isnull().sum())

print("*"*40)

print(test.isnull().sum())
df["Age"].fillna(df.Age.median(),inplace=True)

test["Age"].fillna(test.Age.median(),inplace=True)
df["Embarked"].fillna(df.Embarked.median(),inplace=True)

print(df.isnull().sum())

print("*"*40)

print(test.isnull().sum())
df.corr()["Survived"]
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,square=True,linewidths=.5,cmap="Greens")

plt.show()
X_train = df.drop("Survived", axis=1)

y_train = df["Survived"]

X_test  = test.copy()

X_train.shape, y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

pred = logreg.predict(X_test)

acc_logreg = round(logreg.score(X_train,y_train)*100,2)

acc_logreg
svc = SVC()

svc.fit(X_train,y_train)

pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train,y_train)*100,2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn