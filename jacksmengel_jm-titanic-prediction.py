# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")



# pclass

#print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# sex

#print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# number of siblings / spouses

#print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# number of parents

#print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))



#g = sns.FacetGrid(train_df, col='Survived')

#h = g.map(plt.hist, 'Age', bins=20)

#print(h)



grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()

print(grid)



grid3 = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid3.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid3.add_legend()

print(grid3)



# wrangle data



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

print(test_df.head())

combine = [train_df, test_df]



# convert NaN ages to average age

avg_age = train_df['Age'].mean()

for dataset in combine:

    dataset.loc[(dataset.Age.isnull()),'Age'] = avg_age

combine = [train_df, test_df]



# does name have a title in it?

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

combine = [train_df, test_df]



grid4 = pd.crosstab(train_df['Title'], train_df['Sex'])

print(grid4)



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

combine = [train_df, test_df]

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

combine = [train_df, test_df]

# Drop name feature, not useful anymore

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)    



# replace NULLs with median values for Pclass/gender groupings



# first, take a look at the various groupings in histogram form

grid5 = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid5.map(plt.hist, 'Age', alpha=.5, bins=20)

grid5.add_legend()





# create age bands -> five different buckets in population

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)



# replace Age with bucket

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

combine = [train_df, test_df]

# then, drop AgeBand

train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]



# create combined field of Siblinds and parent/children -> size of family

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

combine = [train_df, test_df]

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



# check if someone is alone

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

combine = [train_df, test_df]

# what's correlation to survival?

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



# create column combining Pclass and Age

for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

combine = [train_df, test_df]

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)



# fill in missing "embarked" values

# C is most common:

freq_port = train_df.Embarked.dropna().mode()[0]



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

combine = [train_df, test_df]

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



# Create numeric Port column (converting alpha values to numbers)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

combine = [train_df, test_df]

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



# Create bucket for fare

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [train_df, test_df]

train_df = train_df.drop(['FareBand'], axis=1)





combine = [train_df, test_df]



# Convert gender to number

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

combine = [train_df, test_df]



# Data is now ready to be plugged into a model!!!



#Use logistic regression

#https://en.wikipedia.org/wiki/Logistic_regression



# first, set up variables to compare

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()



# Example of logistic regression

#logreg = LogisticRegression()

#logreg.fit(X_train, Y_train)

#Y_pred = logreg.predict(X_test)

#acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



# Example of Support Vector Machines

# https://en.wikipedia.org/wiki/Support_vector_machine

#coeff_df = pd.DataFrame(train_df.columns.delete(0))

#coeff_df.columns = ['Feature']

#coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

#coeff_df.sort_values(by='Correlation', ascending=False)

#svc = SVC()

#svc.fit(X_train, Y_train)

#Y_pred = svc.predict(X_test)

#acc_svc = round(svc.score(X_train, Y_train) * 100, 2)



# Example of "K nearest neighbors" or j-NN

#https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

#knn = KNeighborsClassifier(n_neighbors = 3)

#knn.fit(X_train, Y_train)

#Y_pred = knn.predict(X_test)

#acc_knn = round(knn.score(X_train, Y_train) * 100, 2)



# Example of Naive Bayes classifiers

#https://en.wikipedia.org/wiki/Naive_Bayes_classifier

#gaussian = GaussianNB()

#gaussian.fit(X_train, Y_train)

#Y_pred = gaussian.predict(X_test)

#acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



#Example of Perceptron

#https://en.wikipedia.org/wiki/Perceptron

#perceptron = Perceptron()

#perceptron.fit(X_train, Y_train)

#Y_pred = perceptron.predict(X_test)

#acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)



# Example of random forest

# https://en.wikipedia.org/wiki/Random_forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })



submission.to_csv('jsm_submission.csv', index=False)