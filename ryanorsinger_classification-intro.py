import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # sklearn is a machine learning library

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

df["petal_area"] = df.petal_width * df.petal_length

df["sepal_area"] = df.sepal_width * df.sepal_length

df.head()

df.shape
# For our modeling, X is the input variables to build a predictor

X = df.drop(['species'],axis=1)

X.head()
# y is our target variable, we're trying to predict the species of future irises based on their measurements

y = df[['species']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)



X_train.head()
# for classification you can change the algorithm to gini or entropy (information gain).  Default is gini.

# The pattern for sklearn is:

# 1. Make a thing (a new, blank machine learning model of a specific kind)

# 2. Fit the thing (.fitting means to train the machine learning model)

# 3. Use the thing (we'll use our trained model to make predictions on future datapoints)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)

clf
# The easiest part of the entire Data Science pipeline is fitting the machine learning model...

# It's almost anticlimatic...

clf.fit(X_train, y_train)
# Produce a set of species predictions

# Calculate the predicted probability that the prediction is correct

y_pred = clf.predict(X_train)

y_pred_proba = clf.predict_proba(X_train)
labels = sorted(y_train.species.unique())

predicted_labels = [name + " predicted" for name in labels ]



conf = pd.DataFrame(confusion_matrix(y_train, y_pred), index=labels, columns=[predicted_labels])

conf.index.name = "actual"

conf
# Accuracy = total number of (true positives + number of true negatives) divided by the total numbrer of observations

print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))
# The model is a little less accurate on the test data, but 93% accuracy is pretty good!

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
# Actual vs. predicted numbers on the test set!

# y_prediction based on X_test

y_pred = clf.predict(X_test)



labels = sorted(y_train.species.unique())

predicted_labels = [name + " predicted" for name in labels ]



conf = pd.DataFrame(confusion_matrix(y_test, y_pred), index=labels, columns=[predicted_labels])

conf.index.name = "actual"

conf
# The "survived" class is the target we're trying to predict based on other features/columns

titanic = sns.load_dataset("titanic")

titanic.head()