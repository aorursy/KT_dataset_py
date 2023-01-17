# Import all necessary libraries for data analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')
# Read in the data from .csv file

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("Data dimensions of training set: ", train.shape) # Examine the data shape (dimensions)

print("Data dimensions of testing set: ", test.shape) 

train.head(10) # View a certain amount of raw data, default = 5
# Using the describe method we output all statistical information about data

print("Statistical informations of data: ")

train.describe()
# Using .columns method we can examine the features of our dataset

print(train.columns)
# Check for NaN values if any

print(pd.isnull(train).sum())
# Start visualizing each feature and draw connections

sns.barplot(x='Sex', y='Survived', data = train)



# Print percatnage of females survived

print('Percentage of females who survived: ', round(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1] * 100))

# Print percatnage of males survived

print('Percentage of males who survived: ', round(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100))
# Visualize people survived accorinding to the Pclass feature

sns.barplot(x='Pclass', y='Survived', data = train)



# Print percentage of Pclass 1 survivors

print("Percentage of Pclass 1 survivors: ", round(train["Survived"][train['Pclass'] == 1].value_counts(normalize = True)[1] * 100))

# Print percentage of Pclass 2 survivors

print("Percentage of Pclass 2 survivors: ", round(train["Survived"][train['Pclass'] == 2].value_counts(normalize = True)[1] * 100))

# Print percentage of Pclass 3 survivors

print("Percentage of Pclass 3 survivors: ", round(train["Survived"][train['Pclass'] == 3].value_counts(normalize = True)[1] * 100))
# Visualize people survived accorinding to the SibSp feature

sns.barplot(x="SibSp", y="Survived", data = train)



# Print percentage of SibSp survivors

print("Percentage of SibSp equal 0 survivors: ", round(train['Survived'][train['SibSp'] == 0].value_counts(normalize = True)[1] * 100))



# Print percentage of SibSp survivors

print("Percentage of SibSp equal 1 survivors: ", round(train['Survived'][train['SibSp'] == 1].value_counts(normalize = True)[1] * 100))



# Print percentage of SibSp survivors

print("Percentage of SibSp equal 2 survivors: ", round(train['Survived'][train['SibSp'] == 2].value_counts(normalize = True)[1] * 100))
# Visualize people survived accorinding to the Parch feature

sns.barplot(x="Parch", y="Survived", data = train)

plt.show()
# Sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



# Visualize a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
# View statistical information about test data

test.describe(include = 'all')
# Drop the Cabin feature as it is not logical to use and irrelevant 

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
# Drop the Ticket feature as it is also irrelevant to making predictions

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
# Fill in the missing Embarked feature values with mode

print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)
train = train.fillna({'Embarked': 'S'})
# Filling in the Age feature using mode won't work. We will have to predict missing Age values.

# Let's try and use other features to predict our missing Age values.

data = [train, test]



for column in data:

    column['Title'] = column.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

# Extracted the titles of individuals

pd.crosstab(train['Title'], train['Sex'])
# Replace various titles with more common names

for column in data:

    column['Title'] = column['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    column['Title'] = column['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    column['Title'] = column['Title'].replace('Mlle', 'Miss')

    column['Title'] = column['Title'].replace('Ms', 'Miss')

    column['Title'] = column['Title'].replace('Mme', 'Mrs')



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for column in data:

    column['Title'] = column['Title'].map(title_mapping)

    column['Title'] = column['Title'].fillna(0)



train.head()
# Fill missing age with mode Agegroup for each title

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
# Check filled in data

print(pd.isnull(train).sum())
# Map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



train.head()



# Drop the Age feature as we have already created Agegroup

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
# Drop the Name feature as we have Titles already

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
# Map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
# Map each Embarked value to a numerical value

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
# Fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] # Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

        

# Map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



# Drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
# View our updated train dataset

train.head()
# View our updated test dataset

test.head()
# Split the data into training and testing.

from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
# GaussianNB Classifier

from sklearn.naive_bayes import GaussianNB



Naive_Bayes = GaussianNB()

Naive_Bayes.fit(X_train, y_train)

nb_score = round((Naive_Bayes.score(X_test, y_test)) * 100, 2)

print(nb_score)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



LogReg = LogisticRegression()

LogReg.fit(X_train, y_train)

lg_score = round((LogReg.score(X_test, y_test)) * 100, 2)

print(lg_score)
# Support Vector Machine

from sklearn.svm import SVC



SVC = SVC()

SVC.fit(X_train, y_train)

svc_score = round((SVC.score(X_test, y_test)) * 100, 2)

print(svc_score)
# Perceptron

from sklearn.linear_model import Perceptron





perceptron = Perceptron()

perceptron.fit(X_train, y_train)

p_score = round((perceptron.score(X_test, y_test)) * 100, 2)

print(p_score)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier



DecisionTree = DecisionTreeClassifier()

DecisionTree.fit(X_train, y_train)

dt_score = round((DecisionTree.score(X_test, y_test)) * 100, 2)

print(dt_score)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



RandomForest = RandomForestClassifier()

RandomForest.fit(X_train, y_train)

rf_score = round((RandomForest.score(X_test, y_test)) * 100, 2)

print(rf_score)
# K-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



KNN = KNeighborsClassifier()

KNN.fit(X_train, y_train)

knn_score = round((KNN.score(X_test, y_test)) * 100, 2)

print(knn_score)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



SGD = SGDClassifier()

SGD.fit(X_train, y_train)

sgd_score = round((SGD.score(X_test, y_test)) * 100, 2)

print(sgd_score)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



GBC = GradientBoostingClassifier()

GBC.fit(X_train, y_train)

gbc_score = round((GBC.score(X_test, y_test)) * 100, 2)

print(gbc_score)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [svc_score, knn_score, lg_score, rf_score, nb_score, p_score, dt_score,

             sgd_score, gbc_score]})

models.sort_values(by='Score', ascending=False)
# Set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = RandomForest.predict(test.drop('PassengerId', axis=1))



# Set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)