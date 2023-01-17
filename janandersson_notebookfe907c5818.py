import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.tree import export_graphviz

import seaborn as sns

import pydot





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        features = pd.read_csv(os.path.join(dirname, filename))
features.head()
print(features.isnull().sum(axis=0))
features.shape
features.drop('time', axis=1, inplace=True)

features.head()
features.plot(y='age')
features['anaemia'].value_counts().plot(kind='pie', autopct='%1.1f')

plt.ylabel("Count", labelpad=14)

plt.title("Anameia ", y=1.02)
features.plot(y='creatinine_phosphokinase')
features['diabetes'].value_counts().plot(kind='pie', autopct='%1.1f')

plt.ylabel("Count", labelpad=14)

plt.title("Diabetes", y=1.02)
features.plot(y='ejection_fraction')
features['high_blood_pressure'].value_counts().plot(kind='pie', autopct='%1.1f')

plt.ylabel("Count", labelpad=14)

plt.title("High blood pressure", y=1.02)
features.plot(y='platelets')
features.plot(y='serum_creatinine')
features['sex'].value_counts().plot(kind='pie', autopct='%1.1f')

plt.ylabel("Count", labelpad=14)

plt.title("Sex", y=1.02)
features.plot(y='serum_sodium')
features['smoking'].value_counts().plot(kind='pie', autopct='%1.1f')

plt.ylabel("Count", labelpad=14)

plt.title("Smoking", y=1.02)
features['DEATH_EVENT'].value_counts().plot(kind='pie', autopct='%1.1f')

plt.ylabel("Count", labelpad=14)

plt.title("Death event", y=1.02)
X=features.drop(['DEATH_EVENT'], axis=1)

y=features['DEATH_EVENT']
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=1)
print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)
rfc=RandomForestClassifier()

rfc.fit(train_features,train_labels)

p2=rfc.predict(test_features)

s2=accuracy_score(test_labels,p2)

print("Random Forrest Accuracy :", s2*100,'%')
mtx = confusion_matrix(test_labels, p2)

sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cmap="Blues", cbar=False)

plt.ylabel('true label')

plt.xlabel('predicted label')
tree = rfc.estimators_[5]

feature_list = list(features.columns[:-1])



# Pull out one tree from the forest

tree = rfc.estimators_[5]

# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph

(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file

graph.write_png('tree.png')
importances = list(rfc.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];