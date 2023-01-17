# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore") #we close warnings
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
train_df.columns
train_df.describe()
passengerId= test_df['PassengerId']

train_df.info()
train_df['Sex'].value_counts()
a= train_df['Sex'].index
a
b= train_df['Sex'].index.values
b
varValue= train_df["Parch"].value_counts()
varValue
varValue= train_df["Parch"].value_counts()
varValue
plt.bar(varValue.index, varValue) 
plt.xticks(varValue.index, varValue.index.values)

varValue.index
def bar_plot(variable):
    """
     input variable ex: sex, 
     output : bar plot & value count
    """
    #get feature
    var = train_df[variable]
    #count number of categorical variable (value / sample)
    varValue=var.value_counts()

    #visialize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue) 
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    print("{}: \n {}".format(variable, varValue))
category1 = ["Survived", "Sex", "Pclass", "Embarked","SibSp", "Parch"]
for c in category1:
    bar_plot(c)
category2=["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable], bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar=["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)
# Pclass vs Survived
train_df[["Pclass" , "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False) ## we are trying to learn "effect of Pclass" to surviving

train_df[["Sex" , "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)# we are trying to learn "effect of Sex" to surviving
train_df[["SibSp" , "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["Parch" , "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)
def detect_outliers(df, features): ##we are detecting outliers for numerical features
    outlier_indices=[]
    
    for c in features:
        #First Quartile
        Q1 = np.percentile(df[c],25)
        #Third Quartile
        Q3= np.percentile(df[c], 75)
        # IQR
        IQR = Q3-Q1
        # Outlier step
        outlier_step = IQR * 1.5
        #Detect Outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store outliers
        outlier_indices.extend(outlier_list_col) # indexes of outliers are stored in outlier_indices
    outlier_indices= Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2) # if data row has more that one outlier, remove it
    
    
    return multiple_outliers




train_df.loc[detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]
#drop outliers
train_df.drop(detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis=0).reset_index(drop=True)

train_df_len = len(train_df)
train_df= pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
train_df.head()
train_df.columns[train_df.isnull().any()] # which columns has missing value
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by = "Embarked")
plt.show()
# We see fare of embarked:C is close to 80. So we need to fill missing values with C
train_df["Embarked"]= train_df["Embarked"].fillna("C")
train_df[train_df["Fare"].isnull()]
# Pclass can be related to fare bcz of that we will look at Pclass
train_df[train_df["Pclass"]==3]
train_df[train_df["Pclass"]==3]["Fare"]
np.mean(train_df[train_df["Pclass"]==3]["Fare"])
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()]