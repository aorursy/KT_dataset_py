# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.columns
train.head()
train.describe()
test.head()
train.columns
train.shape
train.isnull().sum()
test.describe()
test.columns
test.shape
test.isnull().sum()
#data visualisation
sb.barplot(x="Sex", y="Survived", data=train)
#print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

#print percentages of females vs. males that survive
print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

#Pclass feature


sb.barplot(x='Pclass',y = 'Survived', data = train)
#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#SibSp  feature

sb.barplot(x="SibSp", y="Survived", data=train)
print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 3 who survived:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp = 4 who survived:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)
#Parch feature

sb.barplot(x="Parch", y="Survived", data=train)
print("Percentage of Parch = 0 who survived:", train["Survived"][train["Parch"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of Parch = 1 who survived:", train["Survived"][train["Parch"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Parch = 2 who survived:", train["Survived"][train["Parch"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of Parch = 3 who survived:", train["Survived"][train["Parch"] == 3].value_counts(normalize = True)[1]*100)
#print("Percentage of Parch = 4 who survived:", train["Survived"][train["Parch"] == 4].value_counts(normalize = True)[1]*100)
print("Percentage of Parch = 5 who survived:", train["Survived"][train["Parch"] == 5].value_counts(normalize = True)[1]*100)
#Age feature

mean_value_age = round(train['Age'].mean())
train['Age'].fillna(mean_value_age, inplace=True)
train.head(10)
train.isnull().sum()
bins = [ 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sb.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sb.barplot(x=train["CabinBool"], y="Survived", data=train)
plt.show()
train.columns
train.drop(['Fare'], axis = 1, inplace = True)
train.drop(['Cabin'], axis = 1, inplace = True)
train.drop(['Ticket'], axis = 1, inplace = True)

train.head()
test.drop(['Fare'], axis = 1, inplace = True)
test.drop(['Cabin'], axis = 1, inplace = True)
test.drop(['Ticket'], axis = 1, inplace = True)

test.head()
#Embarked Feature
#now we need to fill in the missing values in the Embarked feature
print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)
#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})
#age prediction
#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()

# fill missing age with mode age group for each title
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
#map each Age to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

#dropping the Age for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)


#Embarked Feature
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
predictors.head()
#importing lib
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Support Vector Machines
from sklearn.svm import SVC
# Linear SVC
from sklearn.svm import LinearSVC
# Perceptron
from sklearn.linear_model import Perceptron
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

#models
gaussian = GaussianNB()
logreg = LogisticRegression()
svc = SVC()
linear_svc = LinearSVC()
perceptron = Perceptron()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
knn = KNeighborsClassifier()
sgd = SGDClassifier()
gbk = GradientBoostingClassifier()

# Training the models
gaussian.fit(x_train, y_train)
logreg.fit(x_train, y_train)
svc.fit(x_train, y_train)
linear_svc.fit(x_train, y_train)
perceptron.fit(x_train, y_train)
decisiontree.fit(x_train, y_train)
randomforest.fit(x_train, y_train)
knn.fit(x_train, y_train)
sgd.fit(x_train, y_train)
gbk.fit(x_train, y_train)
# Testing using the same data
# Gaussian Naive Bayes
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Gaussian Naive Bayes:",acc_gaussian)

# Logistic Regression

y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Logistic Regression:",acc_logreg)

# Support Vector Machines

y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Support Vector Machines:",acc_svc)

# Linear SVC

y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Linear SVC:",acc_linear_svc)

# Perceptron

y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Perceptron:',acc_perceptron)

#Decision Tree

y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Decision Tree:',acc_decisiontree)

# Random Forest

y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Random Forest:',acc_randomforest)

# KNN or k-Nearest Neighbors

y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print('KNN or k-Nearest Neighbors:',acc_knn)

# Stochastic Gradient Descent

y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Stochastic Gradient Descent:',acc_sgd)

# Gradient Boosting Classifier

y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Gradient Boosting Classifier:',acc_gbk)
# The best classifier
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)
