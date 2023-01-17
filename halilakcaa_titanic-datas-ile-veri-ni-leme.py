# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



plt.style.use("seaborn-whitegrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PessengerId = test_data["PassengerId"]
train_data.columns
train_data.tail()
train_data.describe()
train_data.info()
def bar_plot(variable):

    """

    input: variable ex: "Sex"

    output: bar plot & value count

    

    """

    # get feature

    var = train_data[variable]

    

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{} \n {}".format(variable,varValue))
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin","Name","Ticket"]

for c in category2:

    

    print("{} \n".format(train_data[c].value_counts()))
def hist_plot (variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_data[variable], bins =50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with histogram ".format(variable))

    plt.show()
numeric_variable = ["Fare","Age","PassengerId"]

for n in numeric_variable:

    hist_plot(n)
#Pclass vs Survived

train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by ="Survived", ascending = False)
#Parch vs Survived

train_data[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
# Sex vs Survived

train_data[["Sex","Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by = "Survived",ascending = False)
# SibSp vs Survived

train_data[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
def detect_outliers(dframe, features):

    outlier_indices = []

    for c in features:

        # 1st quartile

        Q1 = np.percentile(dframe[c],25)

        # 3rd quartile

        Q3 = np.percentile(dframe[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier Step

        outlier_step = IQR * 1.5

        # Detect outlier and their indices

        outlier_list_column = dframe[(dframe[c]< Q1 - outlier_step) | (dframe[c] > Q3 + outlier_step)].index

        # Store indices

        outlier_indices.extend(outlier_list_column)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers
train_data.loc[detect_outliers(train_data,["Age","SibSp","Parch","Fare"])]
# drop outliers

train_data = train_data.drop(detect_outliers(train_data,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
train_data_length = len(train_data)

train_data = pd.concat([train_data, test_data], axis = 0).reset_index(drop = True)
train_data.head()
train_data.columns[train_data.isnull().any()]
train_data.isnull().sum()
train_data[train_data["Embarked"].isnull()]
train_data.boxplot(column = "Fare", by = "Embarked")

plt.show()
train_data["Embarked"] = train_data["Embarked"].fillna("C")

train_data[train_data["Embarked"].isnull()]
train_data[train_data["Fare"].isnull()]
train_data["Fare"] = train_data["Fare"].fillna(np.mean(train_data[train_data["Pclass"] == 3]["Fare"])) 
train_data[train_data["Fare"].isnull()]