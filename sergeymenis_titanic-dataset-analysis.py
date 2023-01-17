#Environment Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import time
from datetime import date
#Load up the datasets
input_dir = '../input/'
test = pd.read_csv(input_dir + 'test.csv')
train = pd.read_csv(input_dir + 'train.csv')
#Examine data for completeness 
for column in train:
    print(column, len(train)-len(train[train[column].isnull()]))
# Looks like Pclass, Name, Sex, SibSp, Parch, Ticket, Fare, Embarked are good features to play with
# but Ticket number seems to be irrelevant. 
# Final set to examine is Pclass, Name, Sex, SibSp, Parch, Ticket, Fare, Embarked vs. Survived
#Take a look at the data
print("--- Training data looks like:\n\n ",train.head(1),"\n---")
print("--- Test data looks like:\n\n ",test.head(1),"\n---")
print()
print("Nulls in training data:\n",train.isnull().any())
print()
print("Nulls in test data:\n",train.isnull().any())
#Create a mini set to test best approaches
train_mini = train.copy()

#Convert categorical values to integers
train_mini.loc[train_mini['Embarked'] == 'C', 'Embarked'] = 1
train_mini.loc[train_mini['Embarked'] == 'Q', 'Embarked'] = 2
train_mini.loc[train_mini['Embarked'] == 'S', 'Embarked'] = 3

train_mini.loc[train_mini['Sex'] == 'male', 'Sex'] = 1
train_mini.loc[train_mini['Sex'] == 'female', 'Sex'] = 2

#Convert test values to integers
test_mini = test.copy()
test_mini.loc[test_mini['Embarked'] == 'C', 'Embarked'] = 1
test_mini.loc[test_mini['Embarked'] == 'Q', 'Embarked'] = 2
test_mini.loc[test_mini['Embarked'] == 'S', 'Embarked'] = 3

test_mini.loc[test_mini['Sex'] == 'male', 'Sex'] = 1
test_mini.loc[test_mini['Sex'] == 'female', 'Sex'] = 2
#Decision Tree
from sklearn import tree
train_features = train_mini[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
target = train_mini['Survived']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features, target)
#Check the performance
print("Score: ", clf.score(train_features, target))
print("Feature importances: ", clf.feature_importances_)
#Before we predict, deal with nulls in data. Fare is not very important but we'll set it to the median
#of the other fares in the same class
test_mini.loc[test_mini['Pclass'] == 1,'Fare'] = test_mini[test_mini['Pclass'] == 1]['Fare'].fillna(test_mini[test_mini['Pclass'] == 1]['Fare'].median())
test_mini.loc[test_mini['Pclass'] == 2,'Fare'] = test_mini[test_mini['Pclass'] == 2]['Fare'].fillna(test_mini[test_mini['Pclass'] == 2]['Fare'].median())
test_mini.loc[test_mini['Pclass'] == 3,'Fare'] = test_mini[test_mini['Pclass'] == 3]['Fare'].fillna(test_mini[test_mini['Pclass'] == 3]['Fare'].median())
#Predict
test_features = test_mini[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
prediction = clf.predict(test_features)
ids = test_mini['PassengerId'].values
#Format for Kaggle
forKaggle = pd.DataFrame({'PassengerId' : ids, 'Survived' : prediction})
#forKaggle.to_csv('~/local.work/kaggle/titanic/data/prediction_'+date.fromtimestamp(time.time()).strftime("%d_%m_%y")+'.csv', index=False)

