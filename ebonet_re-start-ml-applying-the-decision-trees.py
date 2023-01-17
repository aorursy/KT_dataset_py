import numpy as np

import pandas as pd

import sklearn as sk

from sklearn.model_selection import train_test_split
dataset = pd.read_csv('../input/diabetes.csv')

X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',

'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

Y = dataset[['Outcome']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

Y_train.describe()
Y_test.describe()
from sklearn.tree import DecisionTreeClassifier



# Create the classifier

decision_tree_classifier = DecisionTreeClassifier(random_state = 0)



# Train the classifier on the training set

decision_tree_classifier.fit(X_train, Y_train)



# Evaluate the classifier on the testing set using classification accuracy

decision_tree_classifier.score(X_test, Y_test)
from sklearn import tree



dot_file = tree.export_graphviz(decision_tree_classifier, out_file='tree_a1.dot', 

                                feature_names = list(dataset)[0:-1],

                                class_names = ['healthy', 'ill']) 

print("Accuracy on training set: {:.3f}".format(decision_tree_classifier.score(X_train, Y_train)))

print("Accuracy on test set: {:.3f}".format(decision_tree_classifier.score(X_test, Y_test)))
import graphviz

with open("tree_a1.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)
decision_tree_pruned = DecisionTreeClassifier(random_state = 0, max_depth = 2)



decision_tree_pruned.fit(X_train, Y_train)

decision_tree_pruned.score(X_test, Y_test)
pre_pruned_dot_file = tree.export_graphviz(decision_tree_pruned, out_file='tree_pruned.dot', 

                                feature_names = list(dataset)[0:-1],

                                class_names = ['healthy', 'ill'])

with open("tree_pruned.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)