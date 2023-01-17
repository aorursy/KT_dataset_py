import pandas as pd

import statistics

import matplotlib.pyplot as plt



# Importing dataset

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



train.info()

train.describe()
# Find out missing data

total_null = train.isnull().sum()

percent_of_null = train.isnull().sum()/train.isnull().count()*100

percentage = (round(percent_of_null, 1))

missing_data = pd.concat([total_null, percentage], axis=1, keys=['Total # null entries', '% of null entries'])

missing_data
# Fill out missing values



# Embarked - Fill most common values

train['Embarked'].describe()

dataset = [train, test]

for data in dataset:

    data['Embarked'] = data['Embarked'].fillna('S')    
# Age - Based on name title, place them in their respective age groups

dataset = [train, test]

for data in dataset:

    data['Name_Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Sex'], train['Name_Title'])
for data in dataset:

    data['Name_Title'] = data['Name_Title'].replace('Mlle', 'Miss')

    data['Name_Title'] = data['Name_Title'].replace('Ms', 'Miss')

    data['Name_Title'] = data['Name_Title'].replace('Mme', 'Mrs')

    data['Name_Title'] = data['Name_Title'].replace(['Jonkheer', 'Don'], 'Low_Class')

    data['Name_Title'] = data['Name_Title'].replace(['Dr', 'Rev'], 'Others')

    data['Name_Title'] = data['Name_Title'].replace(['Capt', 'Col', 'Major'], 'Officer')

    data['Name_Title'] = data['Name_Title'].replace(['Countess', 'Lady', 'Sir'], 'Upper_Class')

pd.crosstab(train['Sex'], train['Name_Title'])

train[['Name_Title', 'Survived']].groupby(['Name_Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Officer": 5, "Upper_Class": 6, "Low_Class": 7, "Others": 8}

for data in dataset:

    data['Name_Title'] = data['Name_Title'].map(title_mapping)

    data['Name_Title'] = data['Name_Title'].fillna(0)



train.head()
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)



train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)



train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)



train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x]

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)



train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)

train = train.drop(['PassengerId'], axis = 1)

train.head()
test.head()
from sklearn.model_selection import train_test_split

X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape



from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
import csv



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)

submission