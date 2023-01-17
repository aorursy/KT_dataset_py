# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))







#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

from matplotlib.pyplot import *







%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
train.describe(include='all')
'''Pearson Correlation Matrix

'''



# colormap = plt.cm.RdBu

# plt.figure(figsize=(14,12))

# plt.title('Pearson Correlation of Features', y=1.05, size=15)

# sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

#             square=True, cmap=colormap, linecolor='white', annot=True)
print(

    'Percentage of missing values:\n\n{}'.format(pd.isnull(train).sum() * 100 / len(train)))
# fig, axs = plt.subplots(ncols=3)



# sns.barplot(x='Sex', y='Survived', data=train, ax=axs[0]);

# sns.barplot(x='Pclass', y='Survived', data=train, ax=axs[1])

# sns.boxplot(x='SibSp',y='Survived', data=train, ax=axs[2])



# subplots_adjust(left=0.125, bottom=0.9, right=0.15, top=0.9, wspace=0.2, hspace=0.2)
sns.barplot(x='Sex', y='Survived', data=train);



print("Percentage of females who survived: ",

      train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived: ",

      train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

sns.barplot(x='Pclass', y='Survived', data=train);



print("Percentage of females who survived: ",

      train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived: ",

      train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived: ",

      train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)

#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train)



#I won't be printing individual percent values for all of these.

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 3 who survived:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 4 who survived:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)
sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)



bins = [-1, 0, 5, 12, 18,

        24, 35, 60, np.inf]



labels = ['Unknown', 'Baby', 'Child', 'Teenager',

          'Student', 'Young Adult', 'Adult', 'Senior']



train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival



sns.pointplot(x="AgeGroup", y="Survived", data=train,

             fig_size = (7,5))



plt.show()
test.describe(include="all")
#dropping uneecessary cols

test = test.drop(["Cabin", "Ticket"], axis=1)

train = train.drop(["Cabin", "Ticket"], axis=1)



test.head()
print("Number of people embarked from Southampton:" )

S = train[train["Embarked"] == "S"].shape[0]

print(S)

print("Number of people embarked from Cherbourg:" )

C = train[train["Embarked"] == "C"].shape[0]

print(C)

print("Number of people embarked from Queenstown:" )

Q = train[train["Embarked"] == "Q"].shape[0]

print(Q)

     
#Replacing all missing Embarked values with the most frequent occured port



train = train.fillna({"Embarked": "S"})
train["Embarked"].unique()
combined = [train, test]



# Feature engineering



#Create new feature FamilySize as a combination of SibSp and Parch

for dataset in combined:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in combined:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

# train.head()
for dataset in combined:

    dataset['Title'] = dataset["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train.Title, train.Sex)
for dataset in combined:

    dataset["Title"] = (dataset["Title"]

                                    .replace(['Lady', 'Capt', 'Col', 'Don', 'Dr',

                                             'Major', 'Rev', 'Jonkheer', 'Dona'],

                                           "Rare")

                                   .replace("Mlle", "Miss")

                                   .replace("Ms", "Miss")

                                   .replace("Mme", "Mrs")

                                   .replace(["Countess", "Lady","Sir"], "Royal")

                       )

    

train[["Title", "Survived"]].groupby(["Title"], as_index=True).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,

                 "Master": 4, "Royal": 5, "Rare": 6}



for dataset in combined: 

    dataset["Title"] = (dataset["Title"]

                            .map(title_mapping)

                            .fillna(0))

     

train.head()
#age feature



mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult





age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}







for x in range(len(train["AgeGroup"])):

    if train["AgeGroup"][x] == "Unknown":

        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

        

for x in range(len(test["AgeGroup"])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

        

        



age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)









#train = train.drop(['Age'], axis = 1)

#test = test.drop(['Age'], axis = 1)
#drop the name feature since it contains no more useful information.

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
#map each Embarked value to a numerical value



embarked_mapping = {"S": 1,

                    "C": 2,

                    "Q": 3 }





train["Embarked"] = train["Embarked"].map(embarked_mapping)

test["Embarked"] = test["Embarked"].map(embarked_mapping)



print("Done")
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

#train = train.drop(['Fare'], axis = 1)

#test = test.drop(['Fare'], axis = 1)
train.head()
test.head()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)



target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 3)

print(acc_logreg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})



models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = randomforest.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)