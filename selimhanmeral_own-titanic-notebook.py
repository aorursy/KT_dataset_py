# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



from collections import Counter

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_datas = pd.read_csv("/kaggle/input/titanic/train.csv")

test_datas = pd.read_csv("/kaggle/input/titanic/test.csv")



train_datas.columns
train_datas.head()
train_datas.describe()
test_datas.head()
test_datas.describe()
train_datas.info()
def bar_plot(variable):

    """

        input: variable ex: "Sex"

        output: bar plot & value count

    """

    # get feature

    var = train_datas[variable]

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

    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp", "Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print("{} \n".format(train_datas[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_datas[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Fare", "Age","PassengerId"]

for n in numericVar:

    plot_hist(n)
# Plcass vs Survived

train_datas[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Sex vs Survived

train_datas[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# SibSp vs Survived

train_datas[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# SibSp vs Survived

train_datas[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)
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
train_datas.loc[detect_outliers(train_datas,["Age","SibSp","Parch","Fare"])]
# drop outliers

train_datas = train_datas.drop(detect_outliers(train_datas,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
train_datas[train_datas["Embarked"].isnull()]
train_datas.boxplot(column="Fare",by = "Embarked")

plt.show()
train_datas["Embarked"] = train_datas["Embarked"].fillna("C")

train_datas[train_datas["Embarked"].isnull()]
train_datas[train_datas["Fare"].isnull()]
train_datas["Fare"] = train_datas["Fare"].fillna(np.mean(train_datas[train_datas["Pclass"] == 3]["Fare"]))
train_datas[train_datas["Fare"].isnull()]