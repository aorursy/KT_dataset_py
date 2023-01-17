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

warnings.filterwarnings("ignore") #ignore the warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId=test_df["PassengerId"]
df.columns
df
df.describe()
df.info()
def bar_plot(variable):

    """

    input: Variable ex:Sex

    output: barplot&value count

    """

    #get feature

    var=df[variable]

    #Count of Categorical variable(value/sample) 

    varValue=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))       
category1=["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for c in category1:

    bar_plot(c)
category2=["Cabin","Name","Ticket"]

for c in category2:

    print("{} \n".format(df[c].value_counts()))

        
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(df[variable], bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title(" {} distribution with hist".format(variable))

    plt.show()
numericVar=["Fare", "Age", "PassengerId"]

for n in numericVar:

    plot_hist(n)
# Pclass & Sex - Survived

df[["Pclass", "Survived", "Sex"]].groupby(["Pclass", "Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)

# Sex- Survived

df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)

# SibSp- Survived

df[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)

# Parch- Survived

df[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)

# Parch & SibSp - Survived

df[["Parch","SibSp", "Survived"]].groupby(["Parch", "SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)

#3-0 /0-3/3-2 are 100% survived
def detect_outlier(df,features):

    outlier_indices=[]

    

    for c in features:

        #1st quartile(25%)

        Q1=np.percentile(df[c],25)

        #3rd quartile

        Q3=np.percentile(df[c],75)

        #IQR

        IQR=Q3-Q1

        #Outlier step

        outlier_step=IQR*1.5

        #detect outlier and their indices

        outlier_list_col=df[(df[c]<Q1-outlier_step) | (df[c]>Q3+outlier_step)].index

        #store indices

        outlier_indices.extend(outlier_list_col)

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)

    

    return multiple_outliers

        

    
df.loc[detect_outlier(df,["Age", "SibSp", "Parch", "Fare"])]
#drop outliers

train_df=df.drop(detect_outlier(df,["Age", "SibSp", "Parch", "Fare"]), axis=0).reset_index(drop=True)

df.describe()
train_df.describe()
train_df_len=len(train_df)

train_df=pd.concat([train_df,test_df], axis= 0).reset_index(drop=True)
train_df
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
f1=train_df.Fare==80

f2=train_df.Pclass==1

train_df[f1&f2]
train_df.boxplot(column="Fare", by="Embarked")
train_df["Embarked"]=train_df["Embarked"].fillna("C")
train_df[f1&f2]
train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"]==3]["Fare"])
train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()]