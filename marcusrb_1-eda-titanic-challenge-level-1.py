# Check the versions of libraries



# Python version

import sys

print('Python: {}'.format(sys.version))

# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
# Check the versions of libraries

#!pip install --upgrade pandas

#!pip install --upgrade sklearn
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

from scipy.stats import norm

import matplotlib.pyplot as plt

%matplotlib inline
# restringir los permisos del directorio

#!chmod 600 ~/.kaggle/kaggle.json
#import kaggle

#!kaggle competitions download -c titanic
#!unzip -o titanic.zip
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("kaggle")
# Load dataset train and test

train_titanic = pd.read_csv('../input/titanic/train.csv')

test_titanic = pd.read_csv('../input/titanic/test.csv')



# concat these two datasets, this will come handy while processing the data

#titanic_list =  pd.concat(objs=[train_titanic, test_titanic], axis=0).reset_index(drop=True)

#titanic_list = [train_titanic, test_titanic]
# Check train dataframe structure

train_titanic.info()
# Check test dataframe rows and variables

test_titanic.info()
# Check train dataframe basic stats data

train_titanic.describe()
# Check test dataframe basic stats data

test_titanic.describe()
# Check null and NA values for train dataset

na_values = train_titanic.isna().sum()

# Table of absolute frequency

na_values
# Table of relative frequency

train_titanic.isnull().sum()/len(train_titanic)*100
# Check null and NA values for test dataset

na_values = test_titanic.isna().sum()

# Table of absolute frequency

na_values
# Table of relative frequency

test_titanic.isnull().sum()/len(test_titanic)*100
train_titanic['PassengerId'].head(10)
# Remove PassengerId variable

train_titanic.drop(['PassengerId'], axis=1, inplace=True)
sns.barplot(x="Survived", data=train_titanic)
train_titanic.describe()['Survived']
train_titanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x="Pclass", y="Survived", data=train_titanic)
train_titanic[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
sns.barplot(x="Sex", y="Survived", data=train_titanic)
train_titanic['Sex'] = train_titanic['Sex'] == 'male'

test_titanic['Sex'] = test_titanic['Sex'] == 'male'
train_titanic[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
sns.barplot(x="SibSp", y="Survived", data=train_titanic)
train_titanic[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
sns.barplot(x="Parch", y="Survived", data=train_titanic)
train_titanic['FamilySize'] = train_titanic['SibSp'] + train_titanic['Parch'] + 1

train_titanic[["FamilySize", "Survived"]].groupby(['FamilySize'], as_index=False).mean()
sns.barplot(x="FamilySize", y="Survived", data=train_titanic)
# Apply the same above for test_titanic

test_titanic['FamilySize'] = test_titanic['SibSp'] + test_titanic['Parch'] + 1
train_titanic['IsAlone'] = 0

train_titanic.loc[train_titanic['FamilySize'] == 1, 'IsAlone'] = 1

train_titanic[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
sns.barplot(x="IsAlone", y="Survived", data=train_titanic)
# Apply the same above for test_titanic

test_titanic['IsAlone'] = 0

test_titanic.loc[test_titanic['FamilySize'] == 1, 'IsAlone'] = 1
# We remove Ticket variable in both traing and test dataset

train_titanic.drop(['Ticket'], axis=1, inplace=True)

test_titanic.drop(['Ticket'], axis=1, inplace=True)
train_titanic.head(10)
test_titanic.head(10)
# Check ratio Embarked and Survived variable

train_titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
# Fill by frequency

train_titanic['Embarked'] = train_titanic['Embarked'].fillna('S')
# Check ratio Embarked and Survived variable

train_titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x="Embarked", y="Survived", data=train_titanic)
# Apply the same above for test_titanic

test_titanic['Embarked'] = test_titanic['Embarked'].fillna('S')
sns.distplot(train_titanic['Fare'], fit=norm)
train_titanic['Fare'] = np.log1p(train_titanic['Fare'])

sns.distplot(train_titanic['Fare'], fit=norm)
train_titanic['FareGroup'] = pd.qcut(train_titanic['Fare'], 4, labels=['A', 'B', 'C', 'D'])

train_titanic[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean()
sns.barplot(x="FareGroup", y="Survived", data=train_titanic)
# Apply the same above for test_titanic

test_titanic['Fare'] = np.log1p(test_titanic['Fare'])

test_titanic['FareGroup'] = pd.qcut(test_titanic['Fare'], 4, labels=['A', 'B', 'C', 'D'])
# We remove the variable Fare

train_titanic.drop(['Fare'], axis=1, inplace=True)

test_titanic.drop(['Fare'], axis=1, inplace=True)
train_titanic['InCabin'] = ~train_titanic['Cabin'].isnull()
sns.barplot(x="InCabin", y="Survived", data=train_titanic)

plt.show()
# Apply the same above for test_titanic

test_titanic['InCabin'] = ~test_titanic['Cabin'].isnull()
# Check unique Cabin

pd.unique(train_titanic['Cabin'])
# Count values

train_titanic["Cabin"].value_counts()
#Turning cabin number into Deck

train_titanic["Cabin_Data"] = train_titanic["Cabin"].isnull().apply(lambda x: not x)

test_titanic["Cabin_Data"] = test_titanic["Cabin"].isnull().apply(lambda x: not x)
# Create Deck and Room

train_titanic["Deck"] = train_titanic["Cabin"].str.slice(0,1)

train_titanic["Room"] = train_titanic["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")

train_titanic[train_titanic["Cabin_Data"]]
# Create Deck and Room

test_titanic["Deck"] = test_titanic["Cabin"].str.slice(0,1)

test_titanic["Room"] = test_titanic["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")

test_titanic[test_titanic["Cabin_Data"]]
train_titanic["Deck"] = train_titanic["Deck"].fillna("N")

train_titanic["Room"] = round(train_titanic["Room"].fillna(train_titanic["Room"].mean()),0).astype("int")
test_titanic["Deck"] = test_titanic["Deck"].fillna("N")

test_titanic["Room"] = round(test_titanic["Room"].fillna(test_titanic["Room"].mean()),0).astype("int")
train_titanic
# Check unique Deck

pd.unique(train_titanic['Deck'])
# Check unique Room

pd.unique(train_titanic['Room'])
train_titanic['Room'].describe()
bins = [0, 50, 75, 100, np.inf]

labels = ['r1', 'r2', 'r3', 'r4']

train_titanic['RoomGroup'] = pd.cut(train_titanic["Room"], bins, labels = labels)
test_titanic['RoomGroup'] = pd.cut(test_titanic["Room"], bins, labels = labels)
sns.barplot(x="Deck", y="Survived", data=train_titanic)
sns.barplot(x="RoomGroup", y="Survived", data=train_titanic)
# Remove variables except Deck

train_titanic.drop(["Cabin", "Cabin_Data", "Room"], axis=1, inplace=True, errors="ignore")

test_titanic.drop(["Cabin", "Cabin_Data", "Room"], axis=1, inplace=True, errors="ignore")
train_titanic["Age"] = train_titanic["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train_titanic['AgeGroup'] = pd.cut(train_titanic["Age"], bins, labels = labels)
sns.barplot(x="AgeGroup", y="Survived", data=train_titanic)

plt.show()
# Apply the same above for test_titanic

test_titanic["Age"] = test_titanic["Age"].fillna(-0.5)

test_titanic['AgeGroup'] = pd.cut(train_titanic["Age"], bins, labels = labels)
# We remove the variable Age

train_titanic.drop(['Age'], axis=1, inplace=True)

test_titanic.drop(['Age'], axis=1, inplace=True)
train_titanic['Name'].head(10)
import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



# Apply get_title function

train_titanic['Title'] = train_titanic['Name'].apply(get_title)

test_titanic['Title'] = test_titanic['Name'].apply(get_title)



# Check the results

pd.crosstab(train_titanic['Title'], train_titanic['Sex'])
# Create a categorization on train dataset

train_titanic['Title'] = train_titanic['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_titanic['Title'] = train_titanic['Title'].replace('Mlle', 'Miss')

train_titanic['Title'] = train_titanic['Title'].replace('Ms', 'Miss')

train_titanic['Title'] = train_titanic['Title'].replace('Mme', 'Mrs')



# We create a relative table

train_titanic[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Same above create a categorization on test dataset

test_titanic['Title'] = test_titanic['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_titanic['Title'] = test_titanic['Title'].replace('Mlle', 'Miss')

test_titanic['Title'] = test_titanic['Title'].replace('Ms', 'Miss')

test_titanic['Title'] = test_titanic['Title'].replace('Mme', 'Mrs')
sns.barplot(x="Title", y="Survived", data=train_titanic)

plt.show()
# Remove Name variable

train_titanic.drop(['Name'], axis=1, inplace=True)

test_titanic.drop(['Name'], axis=1, inplace=True)
train_titanic.shape, test_titanic.shape
# Save dataset0 and dataset1 for next step: Modeling

train_titanic.to_csv('/kaggle/working/train_eda.csv', index=False)

test_titanic.to_csv('/kaggle/working/test_eda.csv', index=False)
df1 = pd.read_csv("train_eda.csv")

df1.head(10)
df2 = pd.read_csv("test_eda.csv")

df2.head(10)