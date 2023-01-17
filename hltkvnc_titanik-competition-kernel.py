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

train_df.head()

train_df.describe()
train_df.info()
train_df.info()
def ba_plot(variable):
    
    #get feature
    var = train_df[variable]
    #count number of feature
    var_count = var.value_counts()
 
    #grapichal visualisation of this feature count
    plt.bar(var_count.index,var_count,color="red")
    plt.xticks(var_count.index)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{} \n {}".format(variable,var_count))
#set categorical features
category = [ "Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for each in category:
    ba_plot(each)
#set other categorical features
category2 =["Cabin","Name","Ticket"]

for each in category2:
    print(train_df[each].value_counts())
def hist_plot(variable):
    data = train_df[variable]
    
    plt.figure(figsize=(9,3))
    plt.hist(data,color="red",bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
numerical = ["Fare","Age","PassengerId"]
for each in numerical:
    hist_plot(each)
# Pclass vs Survived
train_df[["Pclass","Survived"]].groupby("Pclass",as_index=False).mean().sort_values(by="Survived",ascending = False)
 
# Sex vs Survived
train_df[["Sex","Survived"]].groupby("Sex",as_index=False).mean().sort_values(by="Survived",ascending = False)
# SibSp vs Survived
train_df[["SibSp","Survived"]].groupby("SibSp",as_index=False).mean().sort_values(by="Survived",ascending = False)
train_df.info()

def detect_outliers(df,features):
    outlier_indices = []
    
    for each in features:
        Q1 = np.percentile(df[each],25)
                              
        # 3rd quartile
        Q3 = np.percentile(df[each],75)
   
        # IQR
        IQR = Q3 - Q1
   
        # Outlier step
        outlier_step = IQR * 1.5
   
    
    
        # detect outlier and their indeces
        outlier_list_col = df[(df[each] < Q1 - outlier_step) | (df[each] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)    
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v < 2)
    
    return multiple_outliers
    
#Detect Outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
#Drop Outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop=True)
train_df_len=len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop = True)
train_df.isnull().sum()
train_df[train_df["Fare"].isnull()]
train_df["Fare"] = train_df.fillna(train_df[train_df["Pclass"] == 3 ]["Fare"].mean())