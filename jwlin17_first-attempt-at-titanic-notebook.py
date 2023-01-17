import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train = train.append(test, sort=False)
train.info()
train['Embarked'].fillna('S', inplace=True)
embarked = pd.get_dummies(train['Embarked'])
embarked_test = pd.get_dummies(test['Embarked'])
embarked.head()
train = pd.concat([train, embarked], axis=1)
test = pd.concat([test, embarked_test], axis=1)
train.head()
train.head()
import re

titles = [re.sub('(.*, )|(\\..*)', '', x) for x in train['Name']]
train['Title'] = titles
from collections import Counter

Counter(titles)
train.groupby(['Title']).mean()
title_survivors = ['the Countess', 'Ms', 'Mme', 'Mlle', 'Lady', 'Mrs', 'Miss', 'Dona']
title_not_survivors = ['Capt', 'Rev', 'Jonkheer']
title_low_survivors = ['Mr', 'Don']
title_average_surv = ['Col', 'Dr', 'Major', 'Master', 'Sir']
train['Survivor Title'] = [int(x in title_survivors) for x in train['Title'] ]
train['Dead Title'] = [int(x in title_not_survivors) for x in train['Title'] ]
train['Low Survivor Title'] = [int(x in title_low_survivors) for x in train['Title'] ]
train['Avg Survivor Title'] = [int(x in title_average_surv) for x in train['Title'] ]
train.head()
train = train.replace('female', 1)
train = train.replace('male', 0)
train.head()
train.info()
train[train['Fare'].isnull()]
grouped = train.groupby(['Embarked']).mean()
grouped
train['Fare'].fillna(grouped['Fare']['S'], inplace=True)
train.info()
train['Cabin Area'] = [str(x)[0] for x in train['Cabin']]
train.groupby('Cabin Area').mean()
survivor_cabins = ['B', 'D', 'E', 'C', 'F']
train['Good Cabin'] = [int(x in survivor_cabins) for x in train['Cabin Area']]
train.head()
train.info()
age_train = train
age_train.head()
age_train = age_train.drop(['Cabin', 'Embarked', 'Name', 'Cabin Area', 'Title','PassengerId', 'Ticket'], axis=1)
age_train.head()
from sklearn.ensemble import RandomForestClassifier
age_train = age_train.drop('Survived', axis=1)
test_age_train = age_train[age_train.isnull().any(axis=1)]
test_age_train = test_age_train.drop('Age', axis=1)
age_train = age_train.dropna(axis=0)
age_train['Fare'] = [float(x) for x in age_train['Fare']]
age_train.head()
from sklearn.model_selection import train_test_split

predictors = ['Fare', 'Parch', "Pclass", "Sex", 'SibSp','C', 'Q', 'S', 'Survivor Title', 
              'Dead Title', 'Low Survivor Title', 'Avg Survivor Title', 'Good Cabin']
X_age_train, X_age_test, Y_age_train, Y_age_test = train_test_split(age_train[predictors], age_train["Age"])
X_age_train = age_train.drop('Age', axis=1)
Y_age_train = age_train['Age']
Y_age_train = [int(x) for x in Y_age_train]

age_train.head()
predictors = ['Fare', 'Parch', "Pclass", "Sex", 'SibSp','C', 'Q', 'S', 'Survivor Title', 'Dead Title', 'Low Survivor Title', 'Avg Survivor Title', 'Good Cabin']
clf = RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             random_state=0)
clf.fit(age_train[predictors], Y_age_train)
prediction = clf.predict(test_age_train)
prediction
test_age_train['Age'] = prediction
test_age_train.head()
age_train = age_train.append(test_age_train, sort=False)
age_train.head()
import math
age_train['Survived'] = [x for x in train['Survived']]
age_train = age_train.replace(1.0, 1)
age_train = age_train.replace(0.0, 0)
age_train.head()
train_set = age_train.dropna(axis=0)
test_set = age_train[age_train.isnull().any(axis=1)]
X_train_set = train_set.drop('Survived', axis=1)
Y_train_set = train_set['Survived']
X_test_set = test_set.drop('Survived', axis=1)
predictors = ['Age', 'Fare', 'Parch', "Pclass", "Sex", 'SibSp','C', 'Q', 'S', 'Survivor Title', 'Dead Title', 'Low Survivor Title', 'Avg Survivor Title', 'Good Cabin']
clf = RandomForestClassifier(n_estimators=4000,
                             criterion='gini',
                             max_depth=40,
                             min_samples_split=10,
                             min_samples_leaf=20,
                             oob_score=True,
                             random_state=0)
clf.fit(X_train_set[predictors], Y_train_set)
print("%.4f" % clf.oob_score_)
prediction = clf.predict(X_test_set)
prediction.size
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
submission.head()
submission.to_csv("submission3.csv", index=False)





