# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid') # plt.style.available

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df.PassengerId

train_df.columns
train_df.head()
train_df.describe()
train_df.info()
var1 = train_df["Embarked"]

var1Value = var1.value_counts()

print(var1Value)

print(var1Value.index)

print(var1Value.index.values)
def bar_plot(x:str):

    """

        input: variable examp: "Sex"

        output: bar plot & value count

    """

    # get feature

    var = train_df[x]

    # count number of categorical variable

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(x)

    plt.show()

    print(f"{x}: \n {varValue}")
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print(f"{train_df[c].value_counts()} \n")
def plotHist(var):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[var], bins=25)

    plt.xlabel(var)

    plt.ylabel("Frequency")

    plt.title(f"{var} distrubution with histogtram")

    plt.show()
numericVar = ["Fare", "Age", "PassengerId"]

for i in numericVar:

    plotHist(i)
# Pclass vs Survived

a = train_df[["Pclass", "Survived"]].groupby("Pclass", as_index=False).mean().sort_values(by="Survived", ascending=False)

a.Survived = [float(f"{i:.2f}") for i in a.Survived] # for setting the digits of Survived values

a
# Sex vs Survived

train_df[["Sex", "Survived"]].groupby("Sex", as_index=False).mean().sort_values(by="Survived", ascending=False)

# SibSp vs Survived

train_df[["SibSp", "Survived"]].groupby("SibSp", as_index=False).mean().sort_values(by="Survived", ascending=False)

# Parch vs Survived

train_df[["Parch", "Survived"]].groupby("Parch", as_index=False).mean().sort_values(by="Survived", ascending=False)

# Age vs Survived

train_df[["Age", "Survived"]].groupby("Survived", as_index=False).mean().sort_values(by="Age", ascending=False)
def detectOutlier(df, features):

    outlier_indices = []

    

    for i in features:

        # 1st quartile

        Q1 = np.percentile(df[i], 25)

        # 3rd quartile

        Q3 = np.percentile(df[i], 75)



        # IQR

        IQR = Q3 - Q1

        # Outlier step

        step = IQR * 1.5

        # detect outlier and their indices

        outlier_list_col = df[(df[i] < Q1 - step) | (df[i] > Q3 + step)].index

        # store indices

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = [i for i, j in outlier_indices.items() if j > 2]

    

    return multiple_outliers
# train_df.describe()

# print(detectOutlier(train_df, ["Age", "SibSp", "Parch", "Fare"]))

train_df.loc[detectOutlier(train_df, ["Age", "SibSp", "Parch", "Fare"])]
# drop outliers

droped_df = train_df.drop(detectOutlier(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis=0).reset_index(drop=True)

droped_df
train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare", by="Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")
# Check the filling process

train_df.loc[[61,829]]
train_df[train_df["Fare"].isnull()]
train_df[train_df["Pclass"] == 3]["Fare"].mean()
train_df["Fare"] = train_df["Fare"].fillna(train_df[train_df["Pclass"] == 3]["Fare"].mean())
# Check the filling process

train_df.loc[[1043]]