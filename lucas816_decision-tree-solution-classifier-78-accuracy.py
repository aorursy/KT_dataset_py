import pandas as pd

IS_LOCAL = False

import os

if(IS_LOCAL):

    PATH="../input/heart.csv"

else:

    PATH="../input/"

print(os.listdir(PATH))



ds = pd.read_csv(PATH+'/heart.csv')

ds.head()
#importing some useful libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



ds.describe()
ds.hist('age')

ds.hist('cp')

ds.hist('trestbps')

ds.hist('chol')

ds.hist('fbs')

ds.hist('restecg')

ds.hist('thalach')

ds.hist('exang')

ds.hist('oldpeak')

ds.hist('slope')

ds.hist('ca')

ds.hist('thal')

ds.hist('target')

matrix = ds.corr()

matrix
targets = ds.pop('target').values

discard_data = ds.pop('fbs')

X = ds.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=.3, random_state=30)





from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()



classifier.fit(X_train, y_train)



classifier.score(X_test, y_test)

y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



import graphviz

from sklearn.tree import export_graphviz



dot_data  = export_graphviz(classifier, feature_names=ds.columns, class_names=['Saudavel', 'Doente'])

graph = graphviz.Source(dot_data)

graph



from joblib import dump, load



dump(classifier, 'classifier_001')



new_classifier = load('classifier_001')



new_classifier.score(X_test, y_test)


