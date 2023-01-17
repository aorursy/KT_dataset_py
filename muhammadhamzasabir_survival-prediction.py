# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
full_data = [train, test]
train.info()
train.loc[:,["Pclass","Survived"]].groupby("Pclass").mean()
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

sns.factorplot("Pclass", "Survived", data=train, ax=ax[0])

ax[0].set_title("Survival rate Vs Pclass")

sns.countplot("Pclass",hue="Survived", data=train, ax=ax[1])

ax[1].set_title("Survival within each Pclass")

plt.close()
train.loc[:, ["Sex", "Survived"]].groupby("Sex").mean()
fig, ax = plt.subplots(1,2, figsize=(15, 7))

sns.factorplot("Sex", "Survived", data=train, ax=ax[0])

ax[0].set_title("Survival rate Vs Sex")

sns.countplot("Sex", hue="Survived", data=train, ax=ax[1])

ax[1].set_title("Survival & Death Vs Sex")

plt.close()

plt.show()
sns.factorplot("Sex", "Survived", col="Pclass", data=train)
train.loc[:, ["Embarked", "Survived"]].groupby("Embarked").mean()
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

sns.factorplot("Embarked", "Survived", data=train, ax=ax[0])

ax[0].set_title("Survival Rate Vs Emabrked")

sns.countplot("Embarked", hue="Survived", data=train, ax=ax[1])

ax[1].set_title("Survival & Death Vs Embarked")

plt.close()

plt.show()
sns.factorplot("Embarked", "Survived", col="Sex", data=train)

plt.show()
sns.factorplot("Embarked", "Survived", col="Pclass", data=train)

plt.show()
def get_title(name):

    title = re.search(" ([A-Za-z]+)\.", name)

    if title:

        return title.group(1)

    else:

        return ""
train["Title"] = train["Name"].apply(get_title)

test["Title"] = test["Name"].apply(get_title)
for dataset in full_data:

    dataset["Title"] = dataset["Title"].replace(["Sir", "Rev", "Capt", "Col", "Countess", "Lady", "Don", "Dr", "Major", 

                                                 "Jonkheer", "Dona"], "Rare")

    dataset["Title"] = dataset["Title"].replace(["Mlle", "Mme"], "Miss")

    dataset["Title"] = dataset["Title"].replace("Ms", "Mrs")
train["Title"].unique(), test["Title"].unique()
train.loc[:, ["Title", "Survived"]].groupby("Title").mean()
sns.factorplot("Title", "Survived", data=train)

plt.show()
for dataset in full_data:

    dataset["FamilyMembers"] = dataset["SibSp"] + dataset["Parch"] + 1
train.loc[:, ["FamilyMembers", "Survived"]].groupby("FamilyMembers").mean()
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

sns.countplot("FamilyMembers", hue="Survived", data=train, ax=ax[0])

ax[0].set_title("Survival & Death within Faimly Member")

sns.factorplot("FamilyMembers", "Survived", data=train, ax=ax[1])

ax[1].set_title("Survival Rate Vs Family Members")

plt.close()

plt.show()
(train["Age"].isnull().sum() + test["Age"].isnull().sum())/(len(train)+len(test))
fill_age = train[["Title", "Age"]].groupby(["Title"]).mean()
train.loc[(train["Age"].isnull()) & (train["Title"]=="Master"), "Age"] = fill_age.loc["Master","Age"]

train.loc[(train["Age"].isnull()) & (train["Title"]=="Miss"), "Age"] = fill_age.loc["Miss","Age"]

train.loc[(train["Age"].isnull()) & (train["Title"]=="Mr"), "Age"] = fill_age.loc["Mr","Age"]

train.loc[(train["Age"].isnull()) & (train["Title"]=="Mrs"), "Age"] = fill_age.loc["Mrs","Age"]

train.loc[(train["Age"].isnull())&(train["Title"]=="Rare"), "Age"] = fill_age.loc["Rare", "Age"]
train["Age"].isnull().sum()
for dataset in full_data:

    dataset["CategoricalAge"] = pd.qcut(dataset["Age"], 5)
train.loc[:, ["CategoricalAge", "Survived"]].groupby(["CategoricalAge"]).mean()
for dataset in full_data:

    dataset["Age_Cat"] = 0

    dataset.loc[dataset["Age"]<=20.0, "Age_Cat"] = 1

    dataset.loc[(dataset["Age"]>20.0)&(dataset["Age"]<=26.0), "Age_Cat"] = 2

    dataset.loc[(dataset["Age"]>26.0)&(dataset["Age"]<=32.368), "Age_Cat"] = 3

    dataset.loc[(dataset["Age"]>32.368)&(dataset["Age"]<=38.0), "Age_Cat"] = 4

    dataset.loc[(dataset["Age"]>38.0)&(dataset["Age"]<=80.0), "Age_Cat"] = 5

    
# Checking mean and median to get the idea either data is skewed or not.

for dataset in full_data:

    print(dataset["Fare"].mean(), dataset["Fare"].median())
for dataset in full_data:

    dataset.loc[dataset["Fare"].isnull(), "Fare"] = dataset["Fare"].median()

    dataset["CategoricalFare"] = pd.qcut(dataset["Fare"], 5)
train.loc[:,["CategoricalFare", "Survived"]].groupby(["CategoricalFare"], as_index=False).mean()
for dataset in full_data:

    dataset["Fare_Cat"] = 0

    dataset.loc[dataset["Fare"]<=7.854, "Fare_Cat"] = 1

    dataset.loc[(dataset["Fare"]>7.854)&(dataset["Fare"]<=10.5), "Fare_Cat"] = 2

    dataset.loc[(dataset["Fare"]>10.5)&(dataset["Fare"]<=21.679), "Fare_Cat"] = 3

    dataset.loc[(dataset["Fare"]>21.679)&(dataset["Fare"]<=39.688), "Fare_Cat"] = 4

    dataset.loc[(dataset["Fare"]>39.688)&(dataset["Fare"]<=512.329), "Fare_Cat"] = 5
# Encoding Sex

for dataset in full_data:

    dataset["Sex"] = dataset["Sex"].astype("category")

    dataset["Sex"] = dataset["Sex"].cat.codes
# Encoding Emabrked & Title

train = pd.get_dummies(train, columns=["Embarked", "Title"], drop_first=True)

test = pd.get_dummies(test, columns=["Embarked", "Title"], drop_first=True)
train.head()
test.head()
plt.figure(figsize=(10, 8))

sns.countplot("Survived", data=train)
plt.figure(figsize=(10,8))

train["Survived"].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", shadow=True)
sns.factorplot("Age_Cat", "Survived", data=train)
sns.factorplot("Pclass", "Survived", data=train)
sns.factorplot("Fare_Cat", "Survived", data=train)
sns.factorplot("Sex", "Survived", col="Pclass", data=train)
drop_cols = ["PassengerId", "Name", "Age","SibSp", "Parch", "Ticket", "Fare", "Cabin", "CategoricalAge", "CategoricalFare"]
train.drop(drop_cols, inplace=True, axis=1)

test.drop(drop_cols, inplace=True, axis=1)
plt.figure(figsize=(15, 8))

sns.heatmap(train.corr(), annot=True, cmap="RdYlGn", annot_kws={"Size":10})
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix
train.shape, test.shape
train_x = train.iloc[:, 1:]

train_y = train.iloc[:, 0]

test_x = test
assert train_x.shape[1] == test_x.shape[1]
model = LogisticRegression()

model.fit(train_x, train_y)

predictions = model.predict(test_x)

model.score(train_x, train_y)
dt_model = DecisionTreeClassifier()

dt_model.fit(train_x, train_y)

predictions = dt_model.predict(test_x)

dt_model.score(train_x, train_y)