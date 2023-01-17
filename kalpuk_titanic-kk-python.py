# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv

from sklearn import tree

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#use pandas to read in train and test

#ensure data is read in correctly

#verify data type is data frame

train=pd.read_csv("../input/train.csv",header=0)

test=pd.read_csv("../input/test.csv",header=0)

print(train.head())

print(test.head())

print(type(train))
print(train.describe())

print(train.shape)
list(train.columns.values)
train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

train["Age"]=train["Age"].fillna(train["Age"].median())

#adding child as parameter

train1["Child"]=0

train1["Child"][train1["Age"]<16]=1

train1["Child"][train1["Age"]>=16]=0
from sklearn.model_selection import train_test_split

train1, test1 = train_test_split(train, test_size = 0.25)

#make sure it returns df

print(type(train1))

print(train1.shape)

print(test1.shape)
from sklearn import tree

#convert sex to float 1 for women and 0 for men

features_one=train1[["Age","Sex"]].values

target=train1["Survived"].values

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one,target))
features_two=train1[["Child","Sex"]].values

my_tree_two = tree.DecisionTreeClassifier()

my_tree_two = my_tree_two.fit(features_two, target)

print(my_tree_two.feature_importances_)

print(my_tree_two.score(features_two,target))
features_three=train1[["Child","Sex","Age"]].values

my_tree_three = tree.DecisionTreeClassifier()

my_tree_three = my_tree_three.fit(features_three, target)

print(my_tree_three.feature_importances_)

print(my_tree_three.score(features_three,target))
print("children under 16", sum(train1["Child"]))
#so the are not many under 16 which accounts for the low importance

#how many kids survived

train1["childsur"]=0

train1["childsur"][(train1["Child"]==1)&(train1["Survived"]==1)]=1

print(sum(train1["childsur"]))
train1.describe()
import matplotlib.pyplot as plt

import seaborn as sns

#borrowing visualisations from journey through titanic

sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)

#not sure what this line of code is doing

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)





#lets look at the original train data rather than train1
train.info()
from sklearn import tree

from sklearn.metrics import accuracy_score

#convert sex to float 1 for women and 0 for men

features_one=train1[["Age","Sex"]].values

target=train1["Survived"].values

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

actual=test1["Survived"].values

test_features=test1[["Age","Sex"]]

predict=my_tree_one.predict(test_features)

print(accuracy_score(actual,predict))





features_three=train1[["Age","Sex","Fare"]].values

test_features_three=test1[["Age","Sex","Fare"]].values

my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_one = my_tree_one.fit(features_three, target)

predict=my_tree_one.predict(test_features_three)

print(accuracy_score(actual,predict))





features_three=train1[["Age","Sex","Fare","SibSp"]].values

test_features_three=test1[["Age","Sex","Fare","SibSp"]].values

my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_one = my_tree_one.fit(features_three, target)

predict=my_tree_one.predict(test_features_three)

print(accuracy_score(actual,predict))

features_three=train1[["Age","Sex","Fare","SibSp","Parch"]].values

test_features_three=test1[["Age","Sex","Fare","SibSp","Parch"]].values

my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_one = my_tree_one.fit(features_three, target)

predict=my_tree_one.predict(test_features_three)

print(accuracy_score(actual,predict))

features_three=train[["Age","Sex","Fare","SibSp","Parch"]].values

target=train["Survived"]

test_features_three=test1[["Age","Sex","Fare","SibSp","Parch"]].values

my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_one = my_tree_one.fit(features_three, target)

predict=my_tree_one.predict(test_features_three)

print(accuracy_score(actual,predict))
test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Fare"]=test["Fare"].fillna(test["Fare"].median())

test["Age"]=test["Age"].fillna(test["Age"].median())

target=train["Survived"]

test.info()

test_features=test[["Age","Sex","Fare","SibSp","Parch"]].values

test_features_three=test[["Age","Sex","Fare","SibSp","Parch"]].values

my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_one = my_tree_one.fit(features_three, target)

predict=my_tree_one.predict(test_features_three)

#print(accuracy_score(actual,predict))

predict.to_csv("../output/my_solution_one.csv", index_label = ["PassengerId"])