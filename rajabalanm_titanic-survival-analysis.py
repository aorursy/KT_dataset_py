# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing Libraries

#Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Ignore warnings

import warnings

warnings.filterwarnings('ignore')
#import train and test CSV files

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



print(train.columns.values)
train.head()
test.head()
#see a summary of the training dataset

train.describe(include = "all")
#check for any null values in train data

print(pd.isnull(train).sum())
#check for any null values in test data

print(pd.isnull(test).sum())
#Finding data type of each column

train.info()
#Draw a bar plot of survival by Pclass

sns.barplot(x="Pclass", y="Survived", data=train)



#Print percentage of people by Pclass that survived

for i in range(0,3):

    print("Percentage of Pclass = ",i+1," who survived:", train["Survived"][train["Pclass"] == i+1].value_counts(normalize = True)[1]*100)
#Draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=train)



#Print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#Draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train)



train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Draw a bar plot for Parch vs. survival

sns.barplot(x="Parch", y="Survived", data=train)



train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.set_style("whitegrid")

grid = sns.FacetGrid(train, col='Survived',size=2.8, aspect=1.5)

grid.map(plt.hist, 'Age', bins=20)

grid.add_legend()

plt.show()
grid = sns.FacetGrid(train, row='Embarked', size=2.5, aspect=1.5)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()

plt.show()
freq_port = train.Embarked.dropna().mode()[0]

freq_port
full_data = [train, test]



for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Extract a title for each Name in the train and test datasets

for dataset in full_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in full_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
#Drop the features which are not going to be used.

train = train.drop(['Ticket', 'Cabin', 'Name'], axis = 1)

test = test.drop(['Ticket', 'Cabin', 'Name'], axis = 1)

full_data = [train, test]
#Map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

for dataset in full_data:

    

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)



train.head()
for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['AgeGroup'] = pd.cut(train['Age'], 5)

print(train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup', ascending=True))



train.head()
for dataset in full_data:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train.head()
train = train.drop(['AgeGroup'], axis=1)

full_data = [train, test]

train.head()
#Fill in missing Fare value in test since only test has a missing value

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

#Create FareBand

train['FareBand'] = pd.qcut(train['Fare'], 4)

train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in full_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

#drop the FareBand

train = train.drop(['FareBand'], axis=1)

full_data = [train, test]



train.head()
#Map each Embarked value to a numerical value

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)



train.head()
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False))
train = train.drop(['Parch', 'SibSp', 'FamilySize',], axis=1)

test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

full_data = [train, test]



train.head()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, random_state = 101)
# KNN or K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_knn)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_logreg)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gaussian)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_svc)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_linear_svc)
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_test)

acc_perceptron = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_perceptron)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_test)

acc_decisiontree = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_sgd)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)
test_pred = svc.predict(test.drop(['PassengerId'], axis=1))
test['Survived'] = test_pred
test.head()
submission = test[['PassengerId','Survived']]

submission.to_csv("submission.csv", index=False)