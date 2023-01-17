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
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId=test["PassengerId"]
train.columns
train.shape
train.head()
train.describe().T
train.info()
def bar_plot(variable):

    """

        input:variable ex:"Sex"

        output:bar plot & value_count

    """

    #count number of categorical variable

    varValue=train[variable].value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

    

    
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2=["Cabin","Name","Ticket"]

for c in category2:

    print("{} \n".format(train[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar=["Fare","Age"]

for c in numericVar:

    plot_hist(c)
#Pclass vs Survived

train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#Sex vs Survived

train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#SibSp vs Survived

train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#Parch vs Survived

train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
def detect_outliers(df,features):

    outlier_indices= []

    

    for c in features:

        #1st quantile

        Q1=np.percentile(df[c],25)

        #3rd quantile

        Q3=np.percentile(df[c],75)

        #IQR

        IQR=Q3-Q1

        #Outlier step

        outlier_step=IQR * 1.5

        #detect outlier and their indeces

        outlier_list_col=df[(df[c]<Q1 - outlier_step) | (df[c]>Q3 + outlier_step)].index

        #store indeces

        outlier_indices.extend(outlier_list_col)

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)

    

    return multiple_outliers
train.loc[detect_outliers(train,["Age","SibSp","Parch","Fare"])]
#drop outliers

train=train.drop(detect_outliers(train,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

train_len=len(train)

train=pd.concat([train,test],axis=0).reset_index(drop=True)
train.head()
train.isnull().values.any()
train.isnull().sum()
train["Embarked"]=train["Embarked"].fillna('empty')

train["Fare"] = train["Fare"].fillna(np.mean(train[train["Pclass"] == 3]["Fare"]))
train.isnull().sum()
list1=["SibSp","Parch","Age","Survived","Fare"]

sns.heatmap(train[list1].corr(),annot=True,fmt=".2f")
sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar",size=5)

plt.show()
sns.factorplot(x="Parch",y="Survived",data=train,kind="bar",size=6)

plt.show()
sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar",size=6)
(sns

.FacetGrid(train,col="Survived")

.map(sns.distplot,"Age",bins=25));
(sns

.FacetGrid(train,col="Survived",row="Pclass")

.map(plt.hist,"Age",bins=25));
(sns

.FacetGrid(train,row="Embarked",size=2)

.map(sns.pointplot,"Pclass","Survived","Sex")

.add_legend());
(sns

.FacetGrid(train,row="Embarked",col="Survived",size=3)

.map(sns.barplot,"Sex","Fare")

.add_legend());
train.isnull().sum()
train[train["Age"].isnull()]
sns.factorplot(x="Sex",y="Age",data=train,kind="box");
sns.factorplot(x="Sex",y="Age",hue="Pclass",data=train,kind="box");
sns.factorplot(x="Parch",y="Age",data=train,kind="box");

sns.factorplot(x="SibSp",y="Age",data=train,kind="box");
train["Sex"]=[1 if i=="male" else 0 for i in train["Sex"]]
sns.heatmap(train[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)
index_nan_age = list(train["Age"][train["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train["Age"][((train["SibSp"] == train.iloc[i]["SibSp"]) &(train["Parch"] == train.iloc[i]["Parch"])& (train["Pclass"] == train.iloc[i]["Pclass"]))].median()

    age_med = train["Age"].median()

    if not np.isnan(age_pred):

        train["Age"].iloc[i] = age_pred

    else:

        train["Age"].iloc[i] = age_med
train[train["Age"].isnull()]
