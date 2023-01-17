# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
import graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Loading data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# Cleaning and preparing data
X_train = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
X_test = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

X_train = X_train.drop("Survived", axis=1)
Y_train = train_df["Survived"]

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())
X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())


# Training the tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
# https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c

# Parameters that we will pass to the library that draws the tree, to make it understandable
features = list(X_test)
target_names = ['not survived', 'survived']
               
# Create DOT data               
dot_data = tree.export_graphviz(decision_tree_mod, out_file = 'tree.dot', feature_names = features, class_names = target_names, precision = 2, filled = True)

# Convert to png using system command hdddrequires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
# Training a tree with modified parameters
decision_tree_mod = tree.DecisionTreeClassifier(max_depth = 5, min_samples_split = 5)
decision_tree_mod = decision_tree_mod.fit(X_train, Y_train)
Y_pred = decision_tree_mod.predict(X_test)
