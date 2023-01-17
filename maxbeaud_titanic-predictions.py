import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



train = pd.read_csv("../input/train.csv");

test = pd.read_csv("../input/test.csv");
# All features

train.columns.values
# Feature data types

train.dtypes
# First 5 rows

train.head()
# Last 5 rows

train.tail()
# Describe numerical features

train.describe()
# Describe categorical features

train.describe(include=['O'])
# Sum of missing values

train.isnull().sum()
train = train.drop(['Ticket','Cabin'], axis=1) 

train = train.dropna();



train["Gender"] = train["Sex"].map({"male": 1, "female": 2})
train.eval("Child = Age < 18", inplace=True)

train["Child"] = train["Child"].map(lambda x: 1 if x else 0)
# Validate assumptions about women, children, and the upper-class.

f, axes = plt.subplots(1, 3)

plt.tight_layout()



# Count of women who survived vs not survived

ax = sns.countplot(x="Sex", hue="Survived", data=train, ax=axes[0])

ax.legend(["no", "yes"])



# Count of women who survived vs not survived

ax = sns.countplot(x="Pclass", hue="Survived", data=train, ax=axes[1])

ax.legend("survived", ["no", "yes"])

ax.set_xticklabels(["lower", "middle", "upper"]);



ax = sns.countplot(x="Child", hue="Survived", data=train, ax=axes[2])

ax.legend(["no", "yes"])

ax.set_xticklabels(["no", "yes"]);