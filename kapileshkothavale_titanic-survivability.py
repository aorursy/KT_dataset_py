# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X_train = pd.read_csv("../input/train.csv")

X_train.count()
fig = plt.figure(figsize =(18,6))
plt.subplot2grid((2,3),(0,0))
X_train.Survived.value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Survived")
plt.subplot2grid((2,3),(0,1))
plt.scatter(X_train.Survived, X_train.Age, alpha = 0.1)
plt.title("Survived wrt Age")
plt.subplot2grid((2,3),(0,2))
X_train.Pclass.value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Class")

plt.subplot2grid((2,3),(1,0), colspan =2)
for x in [1,2,3]:
    X_train.Age[X_train.Pclass==x].plot(kind='kde')
plt.title("Age wrt Class")
plt.legend(("1st","2nd","3rd"))

plt.subplot2grid((2,3),(1,2))
X_train.Embarked.value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Embarked")
plt.show()
fig = plt.figure(figsize =(18,6))
plt.subplot2grid((2,3),(0,0))
X_train.Survived.value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
X_train.Survived[X_train.Sex=='male'].value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Men Survived")

plt.subplot2grid((2,3),(0,2))
X_train.Survived[X_train.Sex =='female'].value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Women Survived")

plt.subplot2grid((2,3),(1,0))
X_train.Sex[X_train.Survived==1].value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Gender of Survived")
plt.subplot2grid((2,3),(1,1), colspan =2)
for x in [1,2,3]:
    X_train.Survived[X_train.Pclass==x].plot(kind='kde')
plt.title("Class wrt Survived")
plt.legend(("1st","2nd","3rd"))

plt.show()
fig = plt.figure(figsize =(18,6))
plt.subplot2grid((2,2),(0,0))
X_train.Survived[(X_train.Sex=='male')& (X_train.Pclass==1)].value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Rich Men Survived")

plt.subplot2grid((2,2),(0,1))
X_train.Survived[(X_train.Sex=='male')& (X_train.Pclass==3)].value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Poor Men Survived")

plt.subplot2grid((2,2),(1,0))
X_train.Survived[(X_train.Sex=='female')& (X_train.Pclass==1)].value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Rich Women Survived")

plt.subplot2grid((2,2),(1,1))
X_train.Survived[(X_train.Sex=='female')& (X_train.Pclass==3)].value_counts(normalize = True).plot(kind ="bar", alpha = 0.5)
plt.title("Poor Women Survived")
trains = pd.read_csv("../input/train.csv")
trains.shape
trains ["hyp"] = 0
trains.loc[trains.Sex == "female", "hyp"]=1
trains["result"]=0
trains.loc[trains.Survived==trains["hyp"], "result"]=1
print (trains["result"].value_counts(normalize = True))
trains["Fare"] = trains["Fare"].fillna(trains["Fare"].median())
W_avg = (trains.loc[trains.Sex=='female']).mean()
print (W_avg.Age)
trains.loc[trains.Sex=='female'] = trains.loc[trains.Sex=='female'].fillna(W_avg.Age)
M_avg = (trains.loc[trains.Sex=='male']).mean()
print(M_avg)
trains.loc[trains.Sex=='male'] = trains.loc[trains.Sex == 'male'].fillna(M_avg.Age)
trains.count()
fig = plt.figure(figsize=(20,8))
plt.subplot2grid((2,3),(0,0))
trains.Survived[(trains.Age<=27) & (trains.Sex=='female')].value_counts(normalize=True).plot(kind='bar', alpha = 0.7)
plt.title("Young Women")

plt.subplot2grid((2,3),(0,1))
trains.Survived[(trains.Age>27) & (trains.Sex=='female')].value_counts(normalize=True).plot(kind='bar', alpha = 0.7)
plt.title("Old Women")

plt.subplot2grid((2,3),(1,0))
trains.Survived[(trains.Age<=27) & (trains.Sex=='male')].value_counts(normalize=True).plot(kind='bar', alpha = 0.7)
plt.title("Young Men")


plt.subplot2grid((2,3),(1,1))
trains.Survived[(trains.Age>27) & (trains.Sex=='male')].value_counts(normalize=True).plot(kind='bar', alpha = 0.7)
plt.title("Old Men")

plt.show()
trains.loc[trains["Sex"]=='male','Sex'] = 0
trains.loc[trains["Sex"]=='female','Sex'] = 1

trains["Embarked"] = trains["Embarked"].fillna("S")
trains.loc[trains["Embarked"]=='S','Embarked'] = 0
trains.loc[trains["Embarked"]=='C','Embarked'] = 1
trains.loc[trains["Embarked"]=='Q','Embarked'] = 2

from sklearn import linear_model
target = trains ["Survived"].values
features = trains[["Pclass","Age","Sex","Parch","Fare","SibSp"]].values
classifier = linear_model.LogisticRegression()
classif = classifier.fit(features,target)
print (classif.score(features,target))
from sklearn.naive_bayes import MultinomialNB

classifier2 = MultinomialNB().fit(features,target)
print(classifier2.score(features,target))
from sklearn.tree import DecisionTreeClassifier
classifier3 = DecisionTreeClassifier(max_leaf_nodes = 5, random_state = 0)
classifier3.fit(features,target)
print(classifier3.score(features,target))