# Marc Desvaux de Marigny
# 16 May 2018

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

# Initial data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Filling in missing data
train.fillna(train['Age'].median(), inplace = True)
train.fillna(train['Embarked'].mode(), inplace = True)

test.fillna(train['Age'].median(), inplace = True)
test.fillna(train['Embarked'].mode(), inplace = True)
test.fillna(train['Fare'].mean(), inplace = True)

# Encoding Categoric Variables
train['Female'] = (train['Sex'] == 'female')*1
train['Male'] = (train['Sex'] == 'male')*1
train.drop('Sex', axis =1, inplace = True)

test['Female'] = (test['Sex'] == 'female')*1
test['Male'] = (test['Sex'] == 'male')*1
test.drop('Sex', axis =1, inplace = True)


# Remove unhelpful columns
train.drop(['Name','Cabin','Ticket', 'Embarked'], axis =1, inplace = True)
test.drop(['Name','Cabin','Ticket', 'Embarked'], axis =1, inplace = True)

# Separate out the predictor and output variables
X_train = train.drop(['PassengerId','Survived'], axis =1)
y_train = train['Survived']
X_test = test.drop('PassengerId', axis =1)

# Make classifier
d_tree = tree.DecisionTreeClassifier()

# Fit classifier
d_tree = d_tree.fit(X_train.values, y_train.values)

# Test classifier on training set
pred_train = d_tree.predict(X_train.values)
print("Training Accuracy:", accuracy_score(y_train, pred_train))

# Test classifier on test set
pred_test = d_tree.predict(X_test)
y_true = pd.read_csv('../input/gender_submission.csv')
    
print("Test Accuracy:",accuracy_score(y_true['Survived'].values, pred_test))

# Output results
test['Survived'] = pred_test
test.to_csv('output.csv', columns = ['PassengerId', 'Survived'], index = False)
