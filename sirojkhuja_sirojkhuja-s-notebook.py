# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sbn

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# plt.style.available
train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.info()
def bar_plot(colName):
    """
    input: colName ex: "Age"
    output: bar plot & value counts
    """
    var = train_df[colName]
    
    colValue = var.value_counts()
    
    plt.figure(figsize = (9,3))
    plt.bar(colValue.index, colValue)
    plt.xticks(colValue.index, colValue.index.values)
    plt.ylabel("Frequency")
    plt.title(colName)
    plt.show()
    print("{}: \n {}".format(colName, colValue))
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:
    bar_plot(c)
category2 = ["Cabin","Name","Ticket"]
for n in category2:
    print("{} \n".format(train_df[n].value_counts()))
def hist_plot(colName):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[colName], bins = 80)
    plt.xlabel(colName)
    plt.ylabel("Frequency")
    plt.title("{} dist with histogram".format(colName))
    plt.show()
numericVar = ["Fare","Age","PassengerId"]
for n in numericVar:
    hist_plot(n)
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean()
# Pclass for Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = True)
# Sex for Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
# SibSp for Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
def det_outliers(df, features):
    outliers_in = []
    
    for feat in features:
        # First Q
        Q1 = np.percentile(df[feat],25)
        # Third Q
        Q3 = np.percentile(df[feat],75)
        # IQR = Q3 - Q1
        IQR = Q3 - Q1
        # Outlier Step as OS = IQR * 1.5
        OS = IQR * 1.5
        # Detect outliers and indices
        outlier_list_col = df[(df[feat] < (Q1 - OS)) | (df[feat] > (Q3 + OS))].index
        # Store indeces
        outliers_in.extend(outlier_list_col)
    
    outliers_in = Counter(outliers_in)
    multiple_outliers = list(i for i, v in outliers_in.items() if v > 2)
    
    return multiple_outliers
        
train_df.loc[det_outliers(train_df, ["Age","SibSp","Fare","Parch"])]
train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
train_df.head(10)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column = "Fare", by = "Embarked")
plt.show()
train_df["Embarked"].fillna("C", inplace = True)
train_df[train_df["Embarked"].isnull()]
train_df.isnull().sum()
train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"] == 3]["Fare"])
train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]), inplace = True)
train_df[train_df["Fare"].isnull()]
train_df.isnull().sum()
