# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_pd_data = pd.read_csv("../input/train.csv")  # return DataFrame

test_pd_data = pd.read_csv("../input/test.csv")
train_pd_data.head(8)

combine = [train_pd_data, test_pd_data]
train_pd_data.describe()
test_pd_data.describe()
test_pd_data.info()
print("overall survivied rate:{}".format(train_pd_data["Survived"].mean()))
print(train_pd_data.columns.values)
train_pd_data.tail()
train_pd_data.describe()
train_pd_data.describe(include=['O'])  # include = 'all' 
train_pd_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# as_index : boolean, default True. For aggregated output, return object with group labels as the index. Only relevant for DataFrame input. as_index=False is effectively “SQL-style” grouped output
train_pd_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_pd_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by="Survived", ascending=False)
g = sns.FacetGrid(train_pd_data, col="Survived")

g.map(plt.hist, "Age", bins=20)
grid = sns.FacetGrid(train_pd_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_pd_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_pd_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_pd_data.shape, test_pd_data.shape, 

      combine[0].shape, combine[1].shape)

train_trimed_pd = train_pd_data.drop(["Ticket", "Cabin"], axis=1)

test_trimed_pd = test_pd_data.drop(["Ticket", "Cabin"], axis=1)

combine = [train_trimed_pd, test_trimed_pd]

"After", train_trimed_pd.shape, test_trimed_pd.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset["Title"] = dataset.Name.str.extract("([A-Za-z]+)\.", expand=True)

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.extract.html

# expand, in the version above is default to be False, which returns Series/Index/DataFrame

# if set to be True, then returns DataFrame

pd.crosstab(train_trimed_pd["Title"], train_trimed_pd["Sex"])
for dataset in combine:

    dataset["Title"] = dataset["Title"].replace(["Capt", "Col", "Countess", "Don", "Dr", "Jonkheer",

                     "Lady", "Major", "Rev","Sir"], "Rare")

    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")

    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

    dataset["Title"] = dataset["Title"].replace("Ms", "Miss")



# please pay attention the replace in dataset even change train_trimed_pd

train_trimed_pd[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()
# the dict is a little bit different

title_map = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}

for dataset in combine:

    dataset["Title"] = dataset["Title"].map(title_map)

    dataset["Title"] = dataset["Title"].fillna(0)

    

train_trimed_pd.head()
train_trimed_pd = train_trimed_pd.drop(["PassengerId", "Name"], axis=1)

# why did the test_trimed_pd not drop PassengerId in the sample I refered?

test_trimed_pd = test_trimed_pd.drop(["Name"], axis=1)

combine = [train_trimed_pd, test_trimed_pd]
for dataset in combine:

    dataset["Sex"] = dataset["Sex"].map({"female": 1, "male": 0}).astype(int)



train_trimed_pd.head()    

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.astype.html

# astype: Cast object to input numpy.dtype Return a copy when copy = True (be really careful with this!)
grid = sns.FacetGrid(train_trimed_pd, row="Pclass", col="Sex", size=2.2, aspect=1.6)

grid.map(plt.hist, "Age", alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset["Sex"] == i) & (dataset["Pclass"] == j + 1)]["Age"].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)            

            age_guess = guess_df.median()

            guess_ages[i, j] = int(age_guess / .5 + .5) * .5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), "Age"] = guess_ages[i, j]

            

    dataset["Age"] = dataset["Age"].astype(int)



train_trimed_pd.head()    

train_trimed_pd["AgeBand"] = pd.cut(train_trimed_pd["Age"], 5)

train_trimed_pd[["AgeBand", "Survived"]].groupby("AgeBand", as_index=False).mean()# .sort_values(by="AgeBand", ascending = True)
for dataset in combine:

    dataset.loc[dataset["Age"] <= 16, "Age"] = 0

    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1

    # the parentheses is indespensible

    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2

    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3

    dataset.loc[(dataset["Age"] > 64), "Age"] = 4



train_trimed_pd.head()    
train_trimed_pd = train_trimed_pd.drop("AgeBand", axis=1)

combine = [train_trimed_pd, test_trimed_pd]

train_trimed_pd.head()
# Parch SibSp means?

for dataset in combine:

    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

    

train_trimed_pd[["FamilySize", "Survived"]].groupby("FamilySize", as_index=False).mean().sort_values(by="Survived", ascending=False)
for dataset in combine:

    dataset["IsAlone"] = 0

    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

    

train_trimed_pd[["IsAlone", "Survived"]].groupby("IsAlone", as_index=False).mean().sort_values(by="Survived", ascending=False)   
train_trimed_pd = train_trimed_pd.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_trimed_pd = test_trimed_pd.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_trimed_pd, test_trimed_pd]



train_trimed_pd.head()
for dataset in combine:

    dataset["Pclass*Age"] = dataset["Pclass"] * dataset["Age"]

    

train_trimed_pd[["Pclass*Age", "Pclass", "Age"]].head(10)

# train_trimed_pd.loc[:, ["Pclass*Age", "Pclass", "Age"]].head(10)

# Purely label-location based indexer for selection by label.
freq_port = train_trimed_pd["Embarked"].dropna().mode()[0]
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_trimed_pd[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset["Embarked"] = dataset["Embarked"].map({"C": 0, "Q": 1, "S": 2})# .astype(int)



train_trimed_pd.head()
train_trimed_pd["Fare"].fillna(train_trimed_pd["Fare"].dropna().median(), inplace=True)

test_trimed_pd["Fare"].fillna(test_trimed_pd["Fare"].dropna().median(), inplace=True)

train_trimed_pd.head()
train_trimed_pd["FareBand"] = pd.qcut(train_trimed_pd["Fare"], 4)

train_trimed_pd[["FareBand", "Survived"]].groupby(["FareBand"], as_index=False).mean().sort_values(by="Survived", ascending=False)
for dataset in combine:

    dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0

    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1

    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2

    dataset.loc[dataset["Fare"] > 31, "Fare"] = 3

    dataset["Fare"] = dataset["Fare"].astype(int)

    

train_trimed_pd = train_trimed_pd.drop(["FareBand"], axis=1)

combine = [train_trimed_pd, test_trimed_pd]



train_trimed_pd.head()
test_trimed_pd.head()
X_train = train_trimed_pd.drop("Survived", axis=1)

Y_train = train_trimed_pd["Survived"]

X_test  = test_trimed_pd.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_trimed_pd.columns.delete(0))

coeff_df.columns = ["Feature"]

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by="Correlation", ascending=False)

# why the result is different?
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc