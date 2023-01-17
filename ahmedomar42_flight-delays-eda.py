%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()

train = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_train.csv.zip")

test = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_test.csv.zip")
train.head()
train.shape
test.shape
sns.countplot(train["dep_delayed_15min"])
plt.figure(figsize=(20,10))

sns.countplot(train["DayofMonth"])
sns.countplot(train["DayOfWeek"])
sns.countplot(train["Month"])
plt.figure(figsize=(15,10))

sns.countplot(train["UniqueCarrier"])
sns.distplot(train["Distance"])
sns.distplot(train["DepTime"])
train["Origin"].nunique()
test["Dest"].nunique()
sns.countplot(train.loc[train["dep_delayed_15min"] == "Y"]["Month"])
sns.countplot(train.loc[train["dep_delayed_15min"] == "Y"]["DayOfWeek"])
sns.distplot(train.loc[train["dep_delayed_15min"] == "Y"]["DepTime"])