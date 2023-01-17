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

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):

    """

    input: variable ex: "Sex"

    output: bar plot & value count

    

    """

    # get feature

    var = train_df[variable]

    #counts number of categorical variable (value/sample)

    varValue = var.value_counts()

    

    #visualize

    plt.figure(figsize = (10,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
def plot_hist(variable):

    """

    Fare, Age and passengerId

    """

    

    

    var = train_df[variable]

    

    #Visualize

    plt.figure(figsize = (10,3))

    plt.hist(var,bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distribution with histogram".format(variable))

    plt.show()

    
numericVar = ["Fare", "Age", "PassengerId"]

for c in numericVar:

    plot_hist(c)
# Pclass vs Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)

# Sex vs Survived

train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)

# SibSp vs Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)

# Parch vs Survived

train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)

# Pclass,Sex vs Survived

train_df[["Pclass","Survived","Sex"]].groupby(["Pclass","Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)

# Embarked vs Survived

train_df[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
# Embarked,Pclass vs Survived 

train_df[["Embarked","Survived","Pclass"]].groupby(["Embarked","Pclass"], as_index = False).mean().sort_values(by = "Survived",ascending = False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for i in features:

        #1st quartile

        

        Q1 = np.percentile(df[i],25)

        

        #3rd quartile

        

        Q3 = np.percentile(df[i],75)

        

        # IQR

        

        IQR = Q3 - Q1

        

        # outlier step

        

        outlier_step = IQR * 1.5

        

        #detect outlier and their indeces

        

        

        outlier_list_col = df[(df[i] < Q1 - outlier_step) | (df[i] > Q3 + outlier_step)].index

        

       

        

        

        #store indeces

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    

    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)

        

    return multiple_outliers

        
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
# drop outliers



train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
a= train_df[(train_df["Age"] > 60) | (train_df["Fare"] >70)].index
a = Counter(a)

a
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df], axis = 0).reset_index(drop = True)
train_df.head()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
trainEM_df = train_df[train_df.Pclass == 1]   #Filter for Pclass =1 due to our missing embarked datas have 1 Pclass

trainEM_df.boxplot(column = 'Fare', by="Embarked")

plt.show()

# Filling Embarked



train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
train_df[(train_df.Embarked == "S") & (train_df.Pclass == 3)].boxplot(column = "Fare")

plt.show()
print(train_df[(train_df.Embarked == "S") & (train_df.Pclass == 3)]["Fare"].mean()) #Filled that empty fare with 13.642

train_df["Fare"] = train_df["Fare"].fillna(train_df[(train_df.Embarked == "S") & (train_df.Pclass == 3)]["Fare"].mean())

train_df[train_df["Fare"].isnull()]
train_df.head()
listem = ["SibSp", "Parch", "Age", "Fare", "Survived"]



plt.figure(figsize= (12,9))

sns.heatmap(train_df[listem].corr(), annot=True, fmt='.2f')

plt.show()
f,ax = plt.subplots(figsize = (10,10))



ax = sns.barplot(x='SibSp', y="Survived", data=train_df)



plt.xlabel("SibSp",fontsize=20)

plt.ylabel("Survival Probability",fontsize=20)

ax.tick_params(labelsize=20)

plt.show()
f,ax = plt.subplots(figsize = (10,10))



ax = sns.barplot(x='Parch', y="Survived", data=train_df)



plt.xlabel("Parch",fontsize=20)

plt.ylabel("Survival Probability",fontsize=20)

ax.tick_params(labelsize=20)

plt.show()
f,ax = plt.subplots(figsize = (10,10))



ax = sns.barplot(x='Pclass', y="Survived", data=train_df)



plt.xlabel("SibSp",fontsize=20)

plt.ylabel("Survival Probability",fontsize=20)

ax.tick_params(labelsize=20)

plt.show()




ax = sns.FacetGrid(train_df, col='Survived',size=5)



ax.map(sns.distplot, 'Age',bins=35)



g = sns.FacetGrid(train_df, col="Survived", row="Pclass", size=3)

g.map(plt.hist,"Age", bins=35)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row="Embarked",size=3)

g.map(sns.pointplot, "Pclass","Survived","Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row="Embarked", col='Survived')

g.map(sns.barplot, "Sex", "Fare")

plt.show()
train_df[train_df["Age"].isnull()]
plt.subplots(figsize=(9,9))

sns.boxplot(data=train_df, x="Sex", y="Age")

plt.show()
plt.subplots(figsize=(7,7))

sns.boxplot(data=train_df, x="Sex", y="Age",hue="Pclass")

plt.show()
plt.subplots(figsize=(9,9))

sns.boxplot(data=train_df, x="Parch", y="Age")

plt.show()
plt.subplots(figsize=(9,9))

sns.boxplot(data=train_df, x="SibSp", y="Age")

plt.show()
sns.heatmap(train_df[["Age","SibSp","Parch","Pclass"]].corr(), annot=True)

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)



for i in index_nan_age:



    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"])&  (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()

    

    age_med = train_df["Age"].median()



    if not np.isnan(age_pred):

         

        train_df["Age"].iloc[i] = age_pred

    else:

         train_df["Age"].iloc[i] = age_med
train_df[train_df["Age"].isnull()]