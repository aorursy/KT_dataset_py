# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

#from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
# Load training data

train = pd.read_csv("../input/train.csv")

train.head()
# Load test data

test = pd.read_csv("../input/test.csv")

test.head()
%matplotlib inline

import seaborn as sns

sns.set()
siblings = train["SibSp"].value_counts()

parentChildren = train["Parch"].value_counts()
print(siblings)

print(parentChildren)
siblings.plot(kind="bar")
parentChildren.plot(kind="bar")
familySize = train["SibSp"] + train["Parch"]

familySize.value_counts().plot(kind="bar")
# Show people's ages when it is not NaN

sns.distplot([age for age in train["Age"] if not np.isnan(age)])
# Add column for family size.

train["FamilySize"] = train["Parch"] + train["SibSp"]

train.head()
# Set mean age where age is missing

meanAge = int(train["Age"].mean())

train["AgeClean"] = train["Age"].apply(lambda t: meanAge if np.isnan(t) else t)

train.head()
# Get gender values

train["Gender"] = train["Sex"].apply(lambda t: 1 if t == "male" else 0)

train.head()
# Get distinct values of passenget class

train["Pclass"].describe()

train["Pclass"].unique()
# Get distinct values of gender

train["Sex"].unique()
train["HasAgeNaN"] = train["Age"].apply(lambda t: 1 if np.isnan(t) else 0)

train.head()
# Get title

import re

regexToGetTitle = r'\w+, (\w+)\..+'

titles = {}

for name in train["Name"]:

    if re.match(regexToGetTitle, name) is not None:

        title = re.match(regexToGetTitle, name).group(1)

        titles[title] = 1 if title not in titles.keys() else titles[title] + 1

        

print(titles)



# Delete keys that have less than 10 items

for key in list(titles):

    if titles[key] < 10:

        del titles[key]

        

print(titles)

titleNames = list(titles.keys())



train["Title"] = train["Name"].apply(lambda t: "UNKNOWN" if re.match(regexToGetTitle, t) is None else re.match(regexToGetTitle, t).group(1))

train["TitleIndex"] = train["Title"].apply(lambda t: -1 if t not in titleNames else titleNames.index(t))

train.head()
# Get embarked location

train["Embarked"].nunique()

train["Embarked"].unique()

train["Embarked"].value_counts()

embarkConverter = {'S': 0, 'C': 1, 'Q': 2}

train["EmbarkedID"] = train["Embarked"].apply(lambda t: -1 if t not in list(embarkConverter.keys()) else embarkConverter[t])

train.head()
correlationColumns = ["FamilySize", "Parch", "SibSp", "AgeClean", "Age", "Gender", "Fare", "Pclass", "Survived", "HasAgeNaN", "TitleIndex", "EmbarkedID"]

correlationData = train[correlationColumns]

correlationData.head()
# sns.pairplot(correlationData, hue="Survived", height=2.5)
sns.catplot(y="AgeClean", x="Survived", data=correlationData, kind="swarm")
# Skipping the mean values filled in to "fix" data

sns.catplot(y="Age", x="Survived", data=correlationData.query("Age != 'NaN'"), kind="swarm")
sns.catplot(y="Fare", x="Survived", data=correlationData, kind="swarm")
sns.countplot(x="Pclass", hue="Survived", data=correlationData)
sns.countplot(x="FamilySize", hue="Survived", data=correlationData)
sns.countplot(x="Parch", hue="Survived", data=correlationData)
sns.countplot(x="SibSp", hue="Survived", data=correlationData)
sns.catplot(x="Pclass", y="Fare", hue="Survived", data=correlationData, kind="swarm")
sns.countplot(x="HasAgeNaN", hue="Survived", data=correlationData)
sns.countplot(x="Gender", hue="Survived", data=correlationData)
sns.countplot(x="TitleIndex", hue="Survived", data=correlationData)
sns.countplot(x="EmbarkedID", hue="Survived", data=correlationData)
def transformData(data):

    """Data transformation function: to be used on both training and test data"""

    

    # Gender

    data["Gender"] = data["Sex"].apply(lambda t: 1 if t == "male" else 0)

    

    # Filling empty age values

    meanAge = round(data["Age"].mean(), 0)

    data["AgeClean"] = data["Age"].apply(lambda age: age if not math.isnan(age) else meanAge)

    

    # Filling empty fare values

    meanFare = round(data["Fare"].mean(), 0)

    data["Fare"] = data["Fare"].apply(lambda t: t if not math.isnan(t) else meanFare)

    

    # Creating IsAgeNaN column

    data["HasAgeNaN"] = data["Age"].apply(lambda age: 1 if math.isnan(age) else 0)

    

    # Getting family size

    data["FamilySize"] = data["Parch"] + data["SibSp"]

    

    # Get title index

    data["Title"] = data["Name"].apply(lambda t: "UNKNOWN" if re.match(regexToGetTitle, t) is None else re.match(regexToGetTitle, t).group(1))

    data["TitleIndex"] = data["Title"].apply(lambda t: -1 if t not in titleNames else titleNames.index(t))

    

    # Get embark ID

    data["EmbarkedID"] = data["Embarked"].apply(lambda t: -1 if t not in list(embarkConverter.keys()) else embarkConverter[t])

    

    # Drop columns

    data = data.drop(columns=["Name", "Ticket", "Cabin","Embarked", "Age", "Title"])

    

    return data



train_new = transformData(train)

test_new = transformData(test)

train_new.head(10)
features = ["Pclass", "Fare", "Gender", "AgeClean", "HasAgeNaN", "FamilySize", "TitleIndex", "EmbarkedID"]

X = train_new[features]

y = train_new["Survived"]
clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(X, y)
test_new["Survived"] = clf.predict(test_new[features])

test_new[["PassengerId", "Survived"]].head()
test_new[["PassengerId", "Survived"]].to_csv("Submission - Random Forest.csv", index=False)
!pip install tpot

from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=10, cv=5, verbosity=2)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
test["Survived"] = tpot.predict(test[features])

test[["PassengerId", "Survived"]].head()
test[["PassengerId", "Survived"]].to_csv("TPOT - Submission.csv", index=False)