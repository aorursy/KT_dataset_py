import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

os.getcwd()
train = pd.read_csv('../input/titanic/train.csv')[['Sex','Pclass','Survived']]

test = pd.read_csv('../input/titanic/test.csv')[['Sex','Pclass']]

gender = pd.read_csv('../input/titanic/gender_submission.csv')
train_x=train.copy()

for col in ['Pclass','Sex']:

    X = pd.get_dummies(train_x[col])

    X = X.drop(X.columns[0], axis=1)

    train_x[X.columns] = X

    train_x.drop(col, axis=1, inplace=True)  # drop the original categorical variable (optional)

#train_x.isna().any()



test_x=test.copy()

for col in ['Pclass','Sex']:

    X = pd.get_dummies(test_x[col])

    X = X.drop(X.columns[0], axis=1)

    test_x[X.columns] = X

    test_x.drop(col, axis=1, inplace=True)  # drop the original categorical variable (optional)

#test.isna().any()
train_y=train['Survived']

train_x=train_x.drop(columns='Survived')

test_y=gender['Survived']
from sklearn.tree import export_graphviz

import graphviz

def plot_decision_tree(clf, feature_names, class_names):

    export_graphviz(clf, out_file="adspy_temp.dot", feature_names=feature_names, class_names=class_names, filled = True, impurity = False)

    with open("adspy_temp.dot") as f:

        dot_graph = f.read()

    return graphviz.Source(dot_graph)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier().fit(train_x,train_y)



print('Decision Tree:')

print('tree train:' + str(tree.score(train_x,train_y)))

print('tree test:' + str(tree.score(test_x,test_y)))



plot_decision_tree(tree,[2,3,'male'],['no','yes'])



#forest = RandomForestClassifier().fit(train_x,train_y)

#print('\nRandom Forest:')

#print('forest train:' + str(forest.score(train_x,train_y)))

#print('forest test:' + str(forest.score(test_x,test_y)))
predicted_survival = tree.predict(test_x)

my_prediction = pd.DataFrame({'PassengerId': gender['PassengerId'], 'Survived': predicted_survival})

my_prediction.to_csv('my_prediction.csv', index=False)