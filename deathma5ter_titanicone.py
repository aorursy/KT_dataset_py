# importing dependencies

import pandas as pd 

import numpy as np

# sklearn libraries to use machine learning model

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

# we will add more dependices as we use all the classification models

# displaying items in the directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# loading data files

train = pd.read_csv("../input/train.csv")

# print(train.head())

train_labels = pd.DataFrame({'Survived':[]})

# Got'em this line solves the below mentioned problem

train_labels['Survived'] = train['Survived']

# train_labels = train_labels.rename(columns = {0 : 'Survived'}, inplace = True)

train_features = train.drop('Survived', axis = 'columns')

# print(train_labels.head())

# weird thing when we print train_labels data it doesnot show the column name for the columns, weird why would that happen

# if somebody figures it please mention it to me, also if you can tell me how to add column name for a table with pandas

# how can i make a table with only one column (with proper name)

# print(train_features.head())

# importing test data

test_features = pd.read_csv("../input/test.csv")

test_labels = pd.DataFrame({'PassengerID': [], 'Survived': []})

train.info()
# Cleaning up the data

# dont need name, ticket, cabin, embarked

# fare can be ignored as it changes because the embarked variable what we need is class

# pclass tells us all we need to know.

train_features = train_features.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 'columns')

test_features = test_features.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 'columns')

print(train_features.head())

train_features.columns
# converting sex value to 0's and 1's

from sklearn import preprocessing

df = train_features.copy()

for x in df.columns:

    print(x)

    if df[x].dtype == object:

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[x].values))

        df[x] = lbl.transform(list(df[x].values))

train_features = df.copy()

df = test_features.copy()

for x in df.columns:

    print(x)

    if df[x].dtype == object:

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[x].values))

        df[x] = lbl.transform(list(df[x].values))

test_features = df.copy()
# now we start using our models

train_features.replace()

test_features = test_features.dropna()

clf = DecisionTreeClassifier()

clf.fit(train_features, train_labels)

test_pred = clf.predict(test_features)

submission = pd.DataFrame({'PassengerId': test_features['PassengerId'], 'Survived': testpred})

submission.to_csv('../output/submission.csv', index = False)