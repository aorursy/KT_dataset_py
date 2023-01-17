# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



#Setting random seed

seed = 10

X, y = make_classification(

n_samples=1000, n_features=100, n_informative=20,

n_clusters_per_class=2,

random_state=11)

X_train, X_test, y_train, y_test = train_test_split(X, y,

random_state=11)

clf = DecisionTreeClassifier(random_state=11)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(classification_report(y_test, predictions))
# Decision Tree on Iris Dataset



import pandas as pd

import graphviz

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import datasets 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Setting random seed.

seed = 10



# Loading Iris dataset.

data = pd.read_csv('../input/iris/Iris.csv')

print(data.head())

# Creating a LabelEncoder and fitting it to the dataset labels.

le = LabelEncoder()

le.fit(data['Species'].values)

# Converting dataset str labels to int labels.

y = le.transform(data['Species'].values)

# Extracting the instances data.

X = data.drop('Species', axis=1).values

# Splitting into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, stratify=y, random_state=seed)



# Creating a DecisionTreeClassifier.

# The criterion parameter indicates the measure used (possible values: 'gini' for the Gini index and

# 'entropy' for the information gain).

# The min_samples_leaf parameter indicates the minimum of objects required at a leaf node.

# The min_samples_split parameter indicates the minimum number of objects required to split an internal node.

# The max_depth parameter controls the maximum tree depth. Setting this parameter to None will grow the

# tree until all leaves are pure or until all leaves contain less than min_samples_split samples.

tree = DecisionTreeClassifier(criterion='gini',

min_samples_leaf=5,

min_samples_split=5,

max_depth=None,

random_state=seed)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))



def plot_tree(tree, dataframe, label_col, label_encoder, plot_title):

    label_names = pd.unique(dataframe[label_col])

    # Obtaining plot data.

    graph_data = export_graphviz(tree, feature_names=dataframe.drop(label_col, axis=1).columns,

    class_names=label_names,filled=True,rounded=True, out_file=None)

    # Generating plot.

    graph = graphviz.Source(graph_data)

    graph.render(plot_title)

    return graph



tree_graph = plot_tree(tree, data, 'Species', le, 'Iris')

tree_graph
import pandas as pd

import graphviz

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import datasets 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Setting random seed.

seed = 10



# Loading Mushroom dataset.

data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

# We drop the 'stalk-root' feature because it is the only one containing missing values.

data = data.drop('stalk-root', axis=1)

# Creating a new DataFrame representation for each feature as dummy variables.

dummies = [pd.get_dummies(data[c]) for c in data.drop('class', axis=1).columns]

# Concatenating all DataFrames containing dummy variables.

binary_data = pd.concat(dummies, axis=1)

# Getting binary_data as a numpy.array.

X = binary_data.values

# Getting the labels.

le = LabelEncoder()

y = le.fit_transform(data['class'].values)

# Splitting the binary dataset into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, stratify=y, random_state=seed)



# Creating a DecisionTreeClassifier.

tree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, min_samples_split=5, max_depth=None,

random_state=seed)

tree.fit(X_train, y_train)



# Prediction and Accuracy

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))



print('DecisionTreeClassifier max_depth: {}'.format(tree.tree_.max_depth))



#What if we fit a decision tree with a smaller depth?

tree = DecisionTreeClassifier(criterion='gini',

min_samples_leaf=5,

min_samples_split=5,

max_depth=3,

random_state=seed)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))



# Appending 'label' column to binary DataFrame.

binary_data['class'] = data['class']

tree_graph = plot_tree(tree, binary_data, 'class', le, 'Mushroom')

tree_graph



# Feature Importance

print("Number of Features :", tree.n_features_,", number of classes :\n",tree.n_classes_)

print("Feature Importance :\n",tree.feature_importances_)
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

#Setting random seed

seed = 10



# Dataset Creation

X, y = make_classification(n_samples=1000, n_features=100, n_informative=20,

n_clusters_per_class=2,random_state=11)



# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=11)



clf = RandomForestClassifier(n_estimators=10, random_state=11)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(classification_report(y_test, predictions))



# Feature Importance

print("Number of Features :", clf.n_features_,", number of classes :\n",clf.n_classes_)

print("Feature Importance :\n",clf.feature_importances_)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



# Dataset creation

X, y = make_classification(

n_samples=1000, n_features=50, n_informative=30,

n_clusters_per_class=3,

random_state=11)



#Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)



#Model Creation

tree_clf = DecisionTreeClassifier(random_state=11)

tree_clf.fit(X_train, y_train)

print('Decision tree accuracy: %s' % tree_clf.score(X_test, y_test))



# When an argument for the base_estimator parameter is not passed, the default DecisionTreeClassifier is used

clf = AdaBoostClassifier(n_estimators=50, random_state=11)

clf.fit(X_train, y_train)

accuracies=[]

accuracies.append(clf.score(X_test, y_test))

plt.title('Ensemble Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Number of base estimators in ensemble')

plt.plot(range(1, 51), [accuracy for accuracy in clf.staged_score(X_test, y_test)])



# Feature Importance

print("Number of Features :", tree_clf.n_features_,", number of classes :\n",tree_clf.n_classes_)

print("Feature Importance :\n",tree_clf.feature_importances_)
