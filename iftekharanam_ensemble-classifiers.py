# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
file_path = '../input/train.csv'
data =pd.read_csv(file_path)
data.head()
features = data.iloc[:, 1:]
labels = data.iloc[:, 0]
features.head()
labels.head()
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
decision_tree.score(x_test, y_test)
random_forest = RandomForestClassifier(n_estimators= 10)
random_forest.fit(x_train, y_train)
random_forest.score(x_test, y_test)
bagging_trees = BaggingClassifier(DecisionTreeClassifier(), n_estimators= 20, max_samples= 1.0)
bagging_trees.fit(x_train, y_train)
bagging_trees.score(x_test, y_test)
logit_regressor = LogisticRegression()
decision_tree_classifier = DecisionTreeClassifier()
svm_classifier = SVC()
voting_classifier = VotingClassifier(estimators=[('lr', logit_regressor), ('dt', decision_tree_classifier),
                                                ('svm', svm_classifier)], voting = 'hard')
voting_classifier.fit(x_train.iloc[1:3000], y_train.iloc[1:3000])
voting_classifier.score(x_test, y_test)

