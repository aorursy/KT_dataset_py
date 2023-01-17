# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

test_Patient = test_df["Patient"]
train_df.tail()
train_df.isna().sum()
train_df.info()
train_df["Sex"] = [1 if i == "Male" else 0 for i in train_df["Sex"]]
from sklearn.preprocessing import LabelEncoder

labelEncoder_Y=LabelEncoder()

train_df.iloc[:,6]=labelEncoder_Y.fit_transform(train_df.iloc[:,6].values)
train_df.head()
train_df.describe()
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Age", "Sex","SmokingStatus","Percent","FVC"]

for n in numericVar:

    plot_hist(n)
# SmokingStatus vs FVC

train_df[["SmokingStatus","FVC"]].groupby(["SmokingStatus"], as_index = False).mean().sort_values(by="FVC",ascending = False)
# Sex vs FVC

train_df[["Sex","FVC"]].groupby(["Sex"], as_index = False).mean().sort_values(by="FVC",ascending = False)
# Age vs FVC

train_df[["Age","FVC"]].groupby(["Age"], as_index = False).mean().sort_values(by="FVC",ascending = False)
# Percent vs FVC

train_df[["Percent","FVC"]].groupby(["Percent"], as_index = False).mean().sort_values(by="FVC",ascending = False)
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
list1 = ["Age", "Sex", "SmokingStatus", "FVC","Percent"]

sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")

plt.show()
g = sns.factorplot(x = "Sex", y = "FVC", data = train_df, kind = "bar", size = 6)

g.set_ylabels("FVC Probability")

plt.show()
g = sns.factorplot(x = "Age", y = "FVC", data = train_df, kind = "bar", size = 7)

g.set_ylabels("FVC Probability")

plt.show()
g = sns.factorplot(x = "SmokingStatus", y = "FVC", data = train_df, kind = "bar", size = 6)

g.set_ylabels("FVC Probability")

plt.show()
g = sns.FacetGrid(train_df, col = "Sex", row = "SmokingStatus", size = 2)

g.map(plt.hist, "Age", bins = 20)

g.add_legend()

plt.show()
sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")

plt.show()
sns.factorplot(x = "Sex", y = "Age", hue = "SmokingStatus",data = train_df, kind = "box")

plt.show()
train_df["SmokingStatus"] = train_df["SmokingStatus"].astype("category")

train_df = pd.get_dummies(train_df, columns= ["SmokingStatus"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.head()