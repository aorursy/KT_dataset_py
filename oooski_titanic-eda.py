# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



plt.style.use("seaborn-whitegrid") # we opened grid.

#plt.style.available --> it shows what else we can use.



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.head()
test_PassengerId = df_test["PassengerId"]

df_train.columns
df_train.head()
df_train.describe()
df_train.info()
def bar_plot(variable):

    """

    input: variable ex: "Sex"

    output: bar plot & value count 

    """

    # get variable

    var = df_train[variable]

    # count number of categorical variable(value/sample)

    varvalue = var.value_counts()

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varvalue.index,varvalue,color="r")

    plt.xticks(varvalue.index)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}:\n{}:".format(variable,varvalue))

    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
def plot_histogram(variable):

    plt.figure(figsize = (9,3))

    plt.hist(df_train[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distirbution with hist".format(variable))

    plt.show()

    

    print("{}:\n{}:".format(variable,df_train[variable].value_counts()))
numericVariables = ["Fare","Age","PassengerId"]

for n in numericVariables:

    plot_histogram(n)
# Pclass - Survived

df_train[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending=False) # I wanted to see Pclass and Survived columns together and I grouped according to Pclass. But it returns

# groupby object so to see result we have to specify what we need to see like mean() or max() or etc.
df_train[df_train["Pclass"]==1]["Pclass"].value_counts()
df_train[df_train["Pclass"]==2]["Pclass"].value_counts()
df_train[df_train["Pclass"]==3]["Pclass"].value_counts()
df_train[df_train["Pclass"]==3]["Survived"].describe()
df_train[df_train["Pclass"]==2]["Survived"].describe()
df_train[df_train["Pclass"]==1]["Survived"].describe()
df_train[["Sex","Survived"]].groupby(["Sex"]).mean()
df_train["Sex"].value_counts()
df_train[["SibSp","Survived"]].groupby(["SibSp"]).mean().sort_values(by="Survived",ascending=True)
df_train[(df_train["SibSp"]==0) & (df_train["Survived"] == 1 )]["Sex"].value_counts()
df_train[(df_train["SibSp"]==0) & (df_train["Survived"] == 1 ) & (df_train["Sex"] == "female")]["Age"].describe()
df_train[["Parch","Survived"]].groupby("Parch").mean()
df_train[(df_train["Parch"] == 0)]["Sex"].value_counts()
df_train[(df_train["Parch"]==0) & (df_train["SibSp"] == 0)]["Sex"].value_counts()
df_train[(df_train["Parch"]==0) & (df_train["SibSp"] == 0)]["Age"].min()
df_train[(df_train["Parch"] == 4) | (df_train["Parch"] == 6)| (df_train["Parch"] == 5)]
for i in df_train[df_train["Sex"] == "male"]["Name"]:

    if "jack" in i.lower():

        print(i)
for i in df_train[df_train["Sex"] == "female"]["Name"]:

    if "rose" in i.lower():

        print(i)
def detect_outliers(df,features):

    outlier_indexes = []

    for c in features:

        # 1st quartile

        

        q1 = np.percentile(df[c],25) # 25 for first quartile

        

        # 3rd quartile

        

        q3 = np.percentile(df[c],75) # 75 for third quartile

        

        

        # IQR

        

        IQR = q3 - q1

        

        

        #Outlier Step

        

        outlier_step = IQR * 1.5

        

        

        

        # detect outlier and their indexes

        

        outlier_col_list = df[(df[c] < q1 - outlier_step) | (df[c] > q3 + outlier_step)].index

        

        

        

        # Store indexes

        

        outlier_indexes.extend(outlier_col_list)

        

        

        

        

    outlier_indexes = Counter(outlier_indexes ) # It counts the values which are in the 'outlier_indexes' and gives us how many different value it has.

    

    multiple_outliers = list( i for i , v in outlier_indexes.items() if v>2) 

    

    return multiple_outliers

        

        
df_train.iloc[detect_outliers(df_train,["Age","SibSp","Parch","Fare"])]
# Drop outliers



df_train = df_train.drop(detect_outliers(df_train,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop = True)
df_train
# Concatenating



df_train_len = len(df_train)



df_train = pd.concat([df_train,df_test],axis=0).reset_index(drop=True)
df_train.head()
df_train.columns[df_train.isnull().any()]
df_train.isnull().sum()
df_train[df_train["Embarked"].isnull()]
df_train.boxplot(column = "Fare", by = "Embarked",figsize = (10,10))

plt.show()
df_train["Embarked"] = df_train["Embarked"].fillna("C")
df_train[df_train["Embarked"].isnull()]
df_train[df_train["Fare"].isnull()]
df_train["Fare"] = df_train["Fare"].fillna(df_train[df_train["Pclass"] == 3]["Fare"].mean())
df_train[df_train["Fare"].isnull()]