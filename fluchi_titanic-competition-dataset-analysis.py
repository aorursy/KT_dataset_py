# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



train = pd.read_csv("../input/train.csv")

train.drop("PassengerId", axis=1, inplace=True)

train_na = train.dropna(axis=0, inplace=False)



print ("Training Columns\n", list(train.columns))

print ("Training set shape\n", train.shape)

print ("Training set shape without NaN values\n", train_na.shape)



numerical_columns = ["Age", "Fare"]

string_columns = ["Name", "Ticket", "Cabin"]

categorical_columns = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Survived"]

label = "Survived"



survived = train[train.Survived == 1]

notSurvived = train[train.Survived == 0]



# Any results you write to the current directory are saved as output.
# describing numerical values

train.drop(categorical_columns, axis=1).describe(include=[np.number])
# describing categorical columns

train.drop(numerical_columns, axis=1).describe()
# using a heat map to get a general vision of a dataset

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(corrmat, vmax=0.8, square=True)
# survived age distribution

f, axes = plt.subplots(1, 2, figsize=(12, 5))

g = sns.distplot(survived.Age.dropna(), bins=10, ax=axes[0])

g = sns.distplot(notSurvived.Age.dropna(), bins=10, ax=axes[1], color="red")
# sex x survived

g = sns.factorplot(x="Sex", col="Survived", data=train.dropna(), kind="count", size=5, aspect=1)
# Pclass x survived

sns.factorplot(x="Pclass", col="Survived", data=train.dropna(), kind="count", size=5, aspect=1)
# Parch x survived

sns.factorplot(x="Parch", col="Survived", data=train.dropna(), kind="count", size=5, aspect=1)