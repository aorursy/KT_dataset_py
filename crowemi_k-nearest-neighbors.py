# Introduction to Machine Learning with Python by Andreas Muller and Sarah Guido
# Chapter 2: K-Nearest Neighbors Examples 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import train_test_split from sci-kit learn 
import mglearn 

from sklearn.model_selection import train_test_split
# import training and test data sets from make_forge dataset
X, y = mglearn.datasets.make_forge()
# split the test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Import the sklearn.neighbors 
from sklearn.neighbors import KNeighborsClassifier
# Instantiate the KNeighborsClassifer with 3 neighbors
clf = KNeighborsClassifier(n_neighbors=3)
# Fit the model fo the training set, storing the data to compute predictions
clf.fit(X_train, y_train)
# Make the prediction on the test data 
print("Test set predictions: {}".format(clf.predict(X_test)))
# Check the accuracy of the model against the test data
print("Test set accuracy: {:2f}".format(clf.score(X_test, y_test)))
# Create subplots for decision boundaries plots 
fig, axes = plt.subplots(1,3,figsize=(10,3))
# Run the model for three different neighbor settings 
for n_neighbors, ax in zip([1,3,9], axes):
    #the fit method returns the object self, so we can instantiate and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    # Use mglearn to plot the decision boundary of current n_neighbor
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)
training_accuracy =[]
testing_accuracy = []
# try n_neighbors from 1 to 10 
neighbors_settings = range(1, 10)

for n_neighbors in neighbors_settings: 
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # append the training accuracy for neighbors
    training_accuracy.append(clf.score(X_train, y_train))
    # append the testing accuracy for neighbors 
    testing_accuracy.append(clf.score(X_test, y_test))
    
# plot the accuracy results 
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, testing_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
