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

warnings.filterwarnings("ignore") #Python tarafından verilen hataları gösterme









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test["PassengerId"]
train.columns
train.head()
train.describe()
train.info()
def bar_plot(variable):

    """

        input: variable ex: "Sex"

        output: bar plot & value count

    """

    # get feature

    var = train[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

 

    
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]



for c in category1:

    bar_plot(c)

    
category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print("{} \n".format(train[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Fare", "Age","PassengerId"]

for n in numericVar:

    plot_hist(n)
# Pclass vs Survived

train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# Sex vs Survived

train[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Sibsp vs Survived

train[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Parch vs Survived

train[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)
#Pclass vs Age

train[["Pclass","Age"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Age",ascending=False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train.loc[detect_outliers(train,["Age","SibSp","Parch","Fare"])]
# drop outliers

train = train.drop(detect_outliers(train,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
train_len=len(train)

train=pd.concat([train,test],axis=0).reset_index(drop=True)

train.head()
train.columns[train.isnull().any()]
# kaç tane oldugunu görelim

train.isnull().sum()
#dataframe de nerede olduguna bakalım

train[train["Embarked"].isnull()]
#datayı fare den yola cıkarak doldurmaya calısalım



train.boxplot(column="Fare",by="Embarked")

plt.show()
train["Embarked"]=train["Embarked"].fillna("C")
train[train["Embarked"].isnull()]
train[train["Fare"].isnull()]
train["Fare"] = train["Fare"].fillna(np.mean(train[train["Pclass"] == 3]["Fare"]))