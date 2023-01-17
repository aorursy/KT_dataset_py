# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.describe(include='all')
train_data.isnull().sum()
test_data.isnull().sum()
import seaborn as sns



sns.barplot(x="Sex", y="Survived", data=train_data)
sns.barplot(x="SibSp", y="Survived", data=train_data)
sns.barplot(x="Pclass", y="Survived", data=train_data)
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#sort the ages into logical categories

train_data["Age"] = train_data["Age"].fillna(-0.5)

test_data["Age"] = test_data["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train_data['AgeGroup'] = pd.cut(train_data["Age"], bins, labels = labels)

test_data['AgeGroup'] = pd.cut(test_data["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train_data)
#Missing values for Embarked Column in training set

train_data[train_data.Embarked.isnull()]
#Filling missing values in Embarked Column

train_data = train_data.fillna({"Embarked": "S"})
train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)

test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
#create a combined group of both datasets

combine = [train_data, test_data]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex'])
#replace various titles with more common names

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} #Replacing title groups with numerical values

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_data.head()
# Filling missing ages with mode age group for each title

mr_age = train_data[train_data["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train_data[train_data["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train_data[train_data["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train_data[train_data["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = train_data[train_data["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train_data[train_data["Title"] == 6]["AgeGroup"].mode() #Adult



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train_data["AgeGroup"])):

    if train_data["AgeGroup"][x] == "Unknown":

        train_data["AgeGroup"][x] = age_title_mapping[train_data["Title"][x]]

        

for x in range(len(test_data["AgeGroup"])):

    if test_data["AgeGroup"][x] == "Unknown":

        test_data["AgeGroup"][x] = age_title_mapping[test_data["Title"][x]]
#Assign each age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train_data['AgeGroup'] = train_data['AgeGroup'].map(age_mapping)

test_data['AgeGroup'] = test_data['AgeGroup'].map(age_mapping)



train_data.head()



train_data = train_data.drop(['Age'], axis = 1)

test_data = test_data.drop(['Age'], axis = 1)
#Drop the name feature because the titles are already extracted.

train_data = train_data.drop(['Name'], axis = 1)

test_data = test_data.drop(['Name'], axis = 1)
#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train_data['Sex'] = train_data['Sex'].map(sex_mapping)

test_data['Sex'] = test_data['Sex'].map(sex_mapping)



train_data.head()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)

test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)



train_data.head()
test_data.head()
#drop Fare values

train_data = train_data.drop(['Fare'], axis = 1)

test_data= test_data.drop(['Fare'], axis = 1)
#Splitting train dataset



from sklearn.model_selection import train_test_split



X = train_data.drop(['Survived', 'PassengerId'], axis=1)

Y = train_data["Survived"]

x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size = 0.22, random_state = 0)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



G = GaussianNB()

G.fit(x_train, y_train)

y_pred = G.predict(x_val)

Accuracy_for_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print("Accuracy_for_gaussian: ",Accuracy_for_gaussian)



# Logistic Regression

from sklearn.linear_model import LogisticRegression



L = LogisticRegression()

L.fit(x_train, y_train)

y_pred = L.predict(x_val)

Accuracy_for_logistic_regression = round(accuracy_score(y_pred, y_val) * 100, 2)

print("Accuracy_for_logistic_regression: ",Accuracy_for_logistic_regression)



# Random Forest

from sklearn.ensemble import RandomForestClassifier



R = RandomForestClassifier()

R.fit(x_train, y_train)

y_pred = R.predict(x_val)

Accuracy_for_Random_forest = round(accuracy_score(y_pred, y_val) * 100, 2)

print("Accuracy_for_Random_forest: ",Accuracy_for_Random_forest)
ID = test_data['PassengerId']

Predictions = L.predict(test_data.drop('PassengerId', axis=1))



output = pd.DataFrame({ 'PassengerId' : ID, 'Survived': Predictions })

output.to_csv('submission.csv', index=False)