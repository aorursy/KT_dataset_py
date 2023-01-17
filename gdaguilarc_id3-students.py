# Dependecies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.metrics import classification_report

import sklearn

import graphviz



pd.options.mode.chained_assignment = None  # default='warn'
# Get the path of input data - KAGGLE

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Create the Classifier

students_classifier = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,

 max_features=None, max_leaf_nodes=None, random_state=0, splitter='best')



# Import Data

data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
# Create average column and determine if student passed or failed

data['passed'] = data['math score'] + data['writing score'] + data['reading score']

data['passed'] = data['passed']/3

data['passed'] = (data['passed']> 59.9).astype(int)

data = shuffle(data)
# Partition of the relevant data columns

split_data = train_test_split(data, test_size=.20)



train_data = split_data[0]

test_data = split_data[1]



train_labels = train_data.passed.tolist()

test_labels = test_data.passed.tolist()



# Normalize Data

train_data.loc[:, "writing score"] = train_data["writing score"] // 10

train_data.loc[:, "math score"] = train_data["math score"] // 10

train_data.loc[:, "reading score"] = train_data["reading score"] // 10





test_data.loc[:, "writing score"] = test_data["writing score"] // 10

test_data.loc[:, "math score"] = test_data["math score"] // 10

test_data.loc[:, "reading score"] = test_data["reading score"] // 10





train_data = train_data.drop(columns = "passed")

test_data = test_data.drop(columns = "passed")

# Format Data

train_data = pd.get_dummies(train_data, drop_first=True)

test_data = pd.get_dummies(test_data, drop_first=True)



print("Columns of sample")

print(train_data.columns)

print("------------- First Row ---------------------")

print(train_data.iloc[0])



# Training

trained = students_classifier.fit(train_data, train_labels)
# Test Data and Accuracy

test_predictions = trained.predict(test_data).tolist()

ID3TestAccuracy = sklearn.metrics.accuracy_score(test_labels, test_predictions)



print('ID3 Testing accuracy: ',ID3TestAccuracy)



train_predictions = trained.predict(train_data).tolist()

ID3TrainAccuracy = sklearn.metrics.accuracy_score(train_labels, train_predictions)



print('ID3 Training accuracy: ',ID3TrainAccuracy)
# Plot with Matplotlib

fig, ax = plt.subplots(figsize=(70, 40))

tree.plot_tree(trained, feature_names=train_data.columns,  max_depth=10, fontsize=18)

plt.show()
# Plot with GraphViz

dot_data = tree.export_graphviz(trained, out_file=None, feature_names=train_data.columns, filled=True, rounded=True)

graph = graphviz.Source(dot_data)

graph.render("Students")

graph