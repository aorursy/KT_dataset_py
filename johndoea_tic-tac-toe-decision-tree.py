import numpy as np

import pandas as pd

from pandas import DataFrame, Series

from IPython.display import Image

from io import StringIO

import pydotplus

from sklearn import preprocessing

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
def plot_decision_tree(clf, features, classes):

    dot_data = StringIO()

    tree.export_graphviz(clf, out_file=dot_data, feature_names=features, class_names=classes, filled=True, rounded=True, special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    return Image(graph.create_png())
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/tictactoe-endgame-dataset-uci/tic-tac-toe-endgame.csv',',')

df
df['V1'],v1 = pd.factorize(df['V1'], sort=True)

df['V2'],v2 = pd.factorize(df['V2'], sort=True)

df['V3'],v3 = pd.factorize(df['V3'], sort=True)

df['V4'],v4 = pd.factorize(df['V4'], sort=True)

df['V5'],v5 = pd.factorize(df['V5'], sort=True)

df['V6'],v6 = pd.factorize(df['V6'], sort=True)

df['V7'],v7 = pd.factorize(df['V7'], sort=True)

df['V8'],v8 = pd.factorize(df['V8'], sort=True)

df['V9'],v9 = pd.factorize(df['V9'], sort=True)

df['V10'],v10 = pd.factorize(df['V10'], sort=True)

[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
class_names = [v10[0], v10[1]]

class_names
df
df.info()
df.describe()
feature_names = ['V1','V2','V3','V4', 'V5', 'V6', 'V7', 'V8', 'V9']

x_train = df[feature_names] # Features

x_train

y_train = df['V10'] # Target

y_train
clf = DecisionTreeClassifier(criterion='entropy')

clf = clf.fit(x_train, y_train)

clf
plot_decision_tree(clf, feature_names, class_names) # clf = Tree; feature_names = features; class_names = classes;
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

[x_train, x_test, y_train, y_test]
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=80) # change this classifier and check the impact

clf = clf.fit(x_train,y_train)

plot_decision_tree(clf, feature_names, class_names)
# use the model to make predictions with the test data

y_pred = clf.predict(x_test)

# how did our model perform?

count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
negative_test = np.array ([2, 2, 1, 2, 1, 0, 1, 0, 0])

positive_test = np.array ([2, 2, 2, 1, 1, 0, 1, 0, 0])

test_group = [negative_test, positive_test]

y_pred = clf.predict(test_group)

y_pred # should give [0, 1]