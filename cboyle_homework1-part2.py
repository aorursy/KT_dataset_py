# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

#Run all scripts in order.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import graphviz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/diabetes.csv')

# correlation graph
corr = df.corr()
sns.heatmap(corr, annot = True)

# drop SkinThickness column from dataset
df = df.drop('SkinThickness', 'columns')



df_glucose = df['Glucose']
df_bp = df['BloodPressure']
df_insulin = df['Insulin']
df_bmi = df['BMI']
df_dpf = df['DiabetesPedigreeFunction']
df_age = df['Age']

# impute dataset values using median
df_glucose.replace(to_replace=0, value=df_glucose.median(), inplace=True)
df_bp.replace(to_replace=0, value=df_bp.median(), inplace=True)
df_insulin.replace(to_replace=0, value=df_insulin.median(), inplace=True)
df_bmi.replace(to_replace=0, value=df_bmi.median(), inplace=True)
df_dpf.replace(to_replace=0, value=df_dpf.median(), inplace=True)
df_age.replace(to_replace=0, value=df_age.median(), inplace=True)

# df[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)
# df = df.dropna()

#True/False ratio
print('True/False ratio', df['Outcome'].sum() / df['Outcome'].count(), '\n')

# split into training and test data
split = train_test_split(df, test_size=.3)
train = split[0]
test = split[1]

outcomes_train = train.Outcome
outcomes_test = test.Outcome

train = train.drop('Outcome', 'columns')
test = test.drop('Outcome', 'columns')






clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(train, outcomes_train)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render()
graph

clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(train, outcomes_train)

dot_data = tree.export_graphviz(clf2, out_file=None)
graph = graphviz.Source(dot_data)
graph.render()
graph
prediction_train = clf.predict(train)
prediction_test = clf.predict(test)

# print(prediction_train, '\n')
# print(outcomes_train.values, '\n')

# print(prediction_test, '\n')
# print(outcomes_test.values, '\n')
# accuracy
print(accuracy_score(outcomes_train, prediction_train))
print(accuracy_score(outcomes_test, prediction_test))

#confusion matrix
print('\nconfusion matrix:\n', confusion_matrix(outcomes_test, prediction_test))
# classification report
print('\nclassification matrix:\n', classification_report(outcomes_test, prediction_test))