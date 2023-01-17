# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns

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
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]
train_df.columns
#test_df.columns

train_df.head()
train_df.info()
def bar_plot(variable):
    """
    input: variable ex: "Age"
    output: bar plot and value count
    """
    var= train_df[variable]
    
    varValue = var.value_counts()
    
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
category = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category:
    bar_plot(c)
category2 = ["Cabin","Name","Ticket"]
for i in category2:
     print("{} \n".format(train_df[i].value_counts()))
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins = 80)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} dist with histogram".format(variable))
    plt.show()
# "Fare", "Age", "PassengerId" --> is numrical variables
numericVar = ["Fare","Age"]
for n in numericVar:
    plot_hist(n)
train_df[["Pclass","Survived"]]
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean()

# Pclass for Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending=False)
# Sex for Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending=False)
# SibSp for Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending=False)
# Parch for Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending=False)
def det_outliers(df,features):
    outliers_in = []
    for c in features:
        # First Q
        Q1= np.percentile(df[c],25)
        # Third Q
        Q3= np.percentile(df[c],75)
        # IQR
        IQR = Q3-Q1
        # Outlier_step
        outlier_step = IQR*1.5
        # Detect Outliers and indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index 
        # Store indices
        outliers_in.extend(outlier_list_col)
    outliers_in = Counter(outliers_in)    
    mult_outliers= list(i for i,v in outliers_in.items() if v >2)
    
    return mult_outliers
train_df.loc[det_outliers(train_df,["Age","SibSp","Fare","Parch"])]
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop = True)
train_df.head(10)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by="Embarked")

        
train_df["Embarked"] = train_df["Embarked"].fillna("C")
#train_df[train_df["Embarked"].isnull()]
train_df.isnull().sum()
train_df[train_df["Fare"].isnull()]
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))

train_df.isnull().sum()