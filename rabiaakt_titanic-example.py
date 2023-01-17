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
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

test_passengerId = data_test["PassengerId"]

data_train.head()
plt.style.available #
data_train.columns
data_train.describe()
data_train.info()
data_train.Embarked.unique()
def bar_plot(variable):

    """Input: variable ex :Sex

       Output: bar plot,value count 

    """

    var = data_train[variable]

    count = var.value_counts()

    plt.figure(figsize=(9,3))

    plt.bar(count.index, count)

    plt.xticks(count.index, count.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable, count))

    
category = ["Survived", "Pclass", "Sex", "Embarked", "SibSp"]

for item in category:

    bar_plot(item)
category2 = ["Cabin", "Name", "Ticket"]

for item in category2:

    print("{} \n".format(data_train[item].value_counts()))
def plot_hist(variable):

    plt.figure(figsize= (9,3))

    plt.hist(data_train[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.show()
numeric = ["Fare", "Age", "PassengerId"]

for num in numeric:

    plot_hist(num)
#Pclass - survived

data_train[["Pclass", "Survived"]].groupby("Pclass" , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train[["Sex", "Survived"]].groupby("Sex" , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train[["Age", "Survived"]].groupby("Age" , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train[["Embarked", "Survived"]].groupby("Embarked" , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train[["SibSp", "Survived"]].groupby("SibSp" , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train[["Parch", "Survived"]].groupby("Parch" , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train[["Pclass", "Sex" , "Survived"]].groupby(["Pclass", "Sex"] , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train[["Parch", "SibSp" , "Survived"]].groupby(["Parch", "SibSp"] , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
data_train.boxplot(column = "Fare")

plt.show()
print(data_train.info())

data_train.boxplot(column = "Age")

plt.show()
def outliers(df, variables):

    outliers_list = []

    

    for i in variables:

        #1st quartile

        Q1 = np.percentile(df[i], 25)

        #3rd quartile

        Q3 = np.percentile(df[i], 75)

        #detect  outliers

        IQR = Q3 - Q1

        #outlier step

        outlier_step = IQR * 1.5

        #outlier and their indexes

        outliers_list_col = df[(df[i] < Q1 - outlier_step) | (df[i] > Q3 + outlier_step) ].index

        #store indeces

        outliers_list.extend(outliers_list_col)

    

    outliers_list = Counter(outliers_list)

    multiple_outliers = list(i for i, v in outliers_list.items() if v > 2)

    return multiple_outliers
data_train.loc[outliers(data_train, ["Age", "SibSp", "Parch", "Fare"])]
#drop outliers

data_train = data_train.drop(outliers(data_train, ["Age", "SibSp", "Parch", "Fare"]),axis = 0).reset_index(drop = True)

data_train.loc[outliers(data_train, ["Age", "SibSp", "Parch", "Fare"])]
data_train_len = len(data_train)

data_train = pd.concat([data_train,data_test], axis = 0).reset_index(drop = True)
data_train.head()
data_train.columns[data_train.isnull().any()]
data_train.isnull().sum()
data_train[data_train["Embarked"].isnull()]
data_train.boxplot(column = "Fare", by = "Embarked")

plt.show()
data_train["Embarked"] = data_train["Embarked"].fillna("C")
data_train[data_train["Embarked"].isnull()]
data_train[data_train["Fare"].isnull()]
data_train.boxplot(column = "Fare", by = "Pclass")

plt.show()
np.mean(data_train[data_train["Pclass"] == 3]["Fare"])
data_train["Fare"] = data_train["Fare"].fillna(np.mean(data_train[data_train["Pclass"] == 3]["Fare"]))
data_train[data_train["Fare"].isnull()]
data_train.columns
listt = ["SibSp", "Parch", "Age", "Fare", "Survived"]

sns.heatmap(data_train[listt].corr(), annot = True, fmt = '.2f', linewidth = 0.5, linecolor = "black")

plt.show()
g = sns.factorplot(x = "SibSp", y = "Survived", data = data_train, kind = 'bar', size = 3)

g.set_ylabels("SibSp - Survived")

plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", data = data_train, kind = 'bar', size = 3)

g.set_ylabels("Survived")

plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived", data = data_train, kind = 'bar', size = 3)

g.set_ylabels("Survived Prob")

plt.show()
g = sns.FacetGrid(data_train, col = 'Survived')

g.map(sns.distplot, 'Age', bins = 25 )

plt.show()
g = sns.FacetGrid(data_train, col = 'Survived', row = 'Pclass', size = 2)

g.map(plt.hist, 'Age', bins = 25)

plt.show()
g = sns.FacetGrid(data_train, row = 'Embarked', size = 2)

g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

g.add_legend()

plt.show()
g = sns.FacetGrid(data_train, row = 'Embarked', col = 'Survived', size = 2.5)

g.map(sns.barplot, 'Sex', 'Fare')

g.add_legend()

plt.show()
data_train[data_train.Age.isnull()]
sns.factorplot(x = 'Sex', y = 'Age', data = data_train, kind = 'box')

plt.show()
sns.factorplot(x = 'Sex', y = 'Age', hue = 'Pclass',data = data_train, kind = 'box')

plt.show()
sns.factorplot(x = 'Parch', y = 'Age',data = data_train, kind = 'box')

plt.show()

sns.factorplot(x = 'SibSp', y = 'Age',data = data_train, kind = 'box')

plt.show()

data_train['Sex'] = [1 if each == 'male' else 0 for each in data_train['Sex']]
sns.heatmap(data_train[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']].corr(), annot = True, fmt = '.2f')

plt.show()
index_nan_age = list(data_train[data_train.Age.isnull()].index)



for i in index_nan_age:

    age_predict = data_train['Age'][((data_train['SibSp'] == data_train.iloc[i]['SibSp']) & (data_train['Parch'] == data_train.iloc[i]['Parch']) & (data_train['Pclass'] == data_train.iloc[i]['Pclass']))].median()

    age_med = data_train.Age.median()

    if not np.isnan(age_predict):

        data_train.Age.iloc[i] = age_predict

    else:

        data_train.Age.iloc[i] = age_med

        
data_train[data_train.Age.isnull()]