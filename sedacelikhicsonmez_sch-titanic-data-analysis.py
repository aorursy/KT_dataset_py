# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid') 
import seaborn as sns
from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")
test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId=test_df["PassengerId"]

train_df.columns
train_df.head()
train_df.describe()
train_df.info()

def bba_plot(variable):    
    #get feature
    var = train_df[variable]
    #count number of categorical variable(value/sample)
    varValue=var.value_counts()
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values) #determine the x axis ticks
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show
    print("{}: \n {}".format(variable,varValue))  
category1=['Survived','Pclass','Sex','SibSp','Parch','Embarked']
for c in category1:
    bba_plot(c)

category2=["Cabin","Name","Ticket"]  # there variables have several values so we evelaute them separately
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show
numericVar=["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)
    

#Pclass sv Survied
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived",ascending=False)
#here we use the as_index=false to put a new indexing instead of Pclass
#Sex sv Survied
train_df[["Sex","Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived",ascending=False) 


#SibSp sv Survied
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived",ascending=False)

#Parch sv Survied
train_df[["Parch","Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived",ascending=False)

def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25) #lower quartile
        # 3rd quartile
        Q3 = np.percentile(df[c],75) #upper quartile
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)  # count the #of indices 
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)  # note: i for i expression exactly mean:print i for i in [1,2,3]
    
    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]

#drop outliers

train_df=train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df.head()
train_df.columns[train_df.isnull().any()] #at which columns there are missing values
train_df.isnull().sum()  #how many missing values
#lets combine previous 2 code, an alternative way to find the # of nulls 
train_df[train_df.columns[train_df.isnull().any()]].isnull().sum()
train_df[train_df["Embarked"].isnull()] 
#we will check the Fare(ticket paid amount) according to Embarked(the port the get in the ship)
#and make a forecast that the missing values of embarked
train_df.boxplot(column="Fare", by="Embarked")
plt.show()

#at the box plot we saw the the Fare value=80 is generally get in the ship from port "C"
train_df["Embarked"]=train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()] 
train_df[train_df["Fare"].isnull()]  
np.mean(train_df[train_df["Pclass"]==3]["Fare"]) 
train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()] 