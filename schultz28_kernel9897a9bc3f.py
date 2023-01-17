# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Steven Schultz

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import graphviz


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv("../input/diabetes.csv")
df = df.drop('SkinThickness', 'columns')
df = df.dropna()

df = df[(df[['Glucose','BloodPressure','Insulin', 'BMI','DiabetesPedigreeFunction','Age']] != 0).all(axis=1)]

split_data = train_test_split(df, test_size=.30)

train_data = split_data[0]
test_data = split_data[1]

outcome_train = train_data.Outcome.tolist()
outcome_test = test_data.Outcome.tolist()

train_data_remove = train_data.drop(columns = "Outcome")
test_data_remove = test_data.drop(columns = "Outcome")

# ID3
clf = sklearn.tree.DecisionTreeClassifier(criterion = "entropy")
clf.fit(train_data_remove, outcome_train)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("Diabetes")

graph
test_array = clf.predict(test_data_remove).tolist()
ID3TestAccuracy = sklearn.metrics.accuracy_score(outcome_test, test_array)

print('ID3 Testing accuracy: ',ID3TestAccuracy)


correlation = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()
train_array = clf.predict(train_data_remove).tolist()
ID3TrainAccuracy = sklearn.metrics.accuracy_score(outcome_train, train_array)

print('ID3 Training accuracy: ',ID3TrainAccuracy)
# C4.5
clf2 = sklearn.tree.DecisionTreeClassifier()

clf2.fit(train_data_remove, outcome_train)

dot_data = tree.export_graphviz(clf2, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("Diabetes")

graph
test_array2 = clf2.predict(test_data_remove).tolist()
CTestAccuracy = sklearn.metrics.accuracy_score(outcome_test, test_array2)

print('C4.5 Testing accuracy: ',CTestAccuracy)

train_array2 = clf2.predict(train_data_remove).tolist()

CTrainAccuracy = sklearn.metrics.accuracy_score(outcome_train, train_array2)

print('C4.5 Training accuracy: ', CTrainAccuracy)
prediction_test = clf.predict(test_data_remove).tolist()
print("TESTING DATA")
print('Confusion Matrix with Testing Data with ID3 Algorithm:')
print(confusion_matrix(outcome_test, prediction_test))
print()
print('Classification Report ID3 Algorithm:')
print(classification_report(outcome_test, prediction_test))
print()
print()
print("TRAINING DATA")
prediction_train = clf.predict(train_data_remove).tolist()

print('Confusion Matrix with Training Data with ID3 Algorithm:')
print(confusion_matrix(outcome_train, prediction_train))
print()
print('Classification Report ID3 Algorithm:')
print(classification_report(outcome_train, prediction_train))
prediction_train2 = clf2.predict(train_data_remove).tolist()

print("TESTING DATA")
print('Confusion Matrix with Testing Data with C4.5 Algorithm: ')
print(confusion_matrix(outcome_train, prediction_train2))
print()
print('Classification Report C4.5 Algorithm:')
print(classification_report(outcome_train, prediction_train2))
print()
print()
print("TRAINING DATA")
print('Confusion Matrix with Training Data with C4.5 Algorithm:')
print(confusion_matrix(outcome_train, prediction_train2))
print()
print('Classification Report C4.5 Algorithm:')
print(classification_report(outcome_train, prediction_train2))
