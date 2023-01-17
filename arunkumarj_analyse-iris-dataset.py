# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import sys, os

import numpy

import scipy

import matplotlib

import sklearn

import pandas



from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from pandas.tools.plotting import andrews_curves



URL = "../input/Iris.csv"

print ("Dataset URL: ", URL)

dataset = pandas.read_csv(URL)
print ("Display column names")

print (dataset.columns.values)

print ("")
print ("Display the count of dataset")

print (dataset.shape)

print ("")
print ("Display first 20 values")

print (dataset.head(20))

print ("")
print ("Description of dataset")

print (dataset.describe())

print ("")
print ("Group the dataset based on the class")

print (dataset.groupby("Species").size())

print ("")
print ("Data types of the dataset")

print (dataset.dtypes)

print ("")
new_ds = dataset.drop('Id', 1)

print (new_ds.dtypes)



#display the dataset values in basic chart

new_ds.plot(kind="line", subplots=True, sharex=False, sharey=False)

plt.show()
#display in histogram

new_ds.hist()

plt.show()
new_ds.plot(x='SepalLengthCm', y='SepalWidthCm', style='o')

plt.show()
#using andrews curves

andrews_curves(new_ds, "Species")

plt.show()
colors= {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}

scatter_matrix(new_ds, c=new_ds["Species"].apply(lambda x: colors[x]))

plt.show()