# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris_original = pd.read_csv("../input/Iris.csv")
iris = iris_original.copy()
iris.head()
iris.Species.value_counts()
iris[iris.Species=='Iris-setosa'].SepalLengthCm
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='blue', label='Versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='green', label='Virginica', ax=fig)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='Versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='Virginica', ax=fig)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

iris.hist()
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
plt.figure(figsize=(15,10))
plt.subplot(221)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(222)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(223)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(224)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
g = sns.lmplot(x="SepalWidthCm", y="SepalLengthCm", hue="Species", data=iris)
g = sns.lmplot(x="PetalWidthCm", y="PetalLengthCm", hue="Species", data=iris)
plt.figure(figsize=(15,10))
plt.subplot(221)
sns.boxplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(222)
sns.boxplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(223)
sns.boxplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(224)
sns.boxplot(x='Species',y='SepalWidthCm',data=iris)
iris.head()
iris = iris.drop('Id',axis=1)
matrix = iris.corr()
f, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="cubehelix_r")
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
X_train = iris.drop('Species', axis=1)
y_train = iris.Species
train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, random_state = 0)
my_svm_model = svm.SVC(kernel='linear')
my_svm_model.fit(train_X, train_y)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_svm_model, test_X, test_y, cv=kfold)
print("SVM Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
model = LogisticRegression()
model.fit(train_X,train_y)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, test_X, test_y, cv=kfold)
print("Logistic Regression Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
model=DecisionTreeClassifier()
model.fit(train_X,train_y)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, test_X, test_y, cv=kfold)
print("Decision Tree Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
model.fit(train_X,train_y)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, test_X, test_y, cv=kfold)
print("K-Nearest neighbors Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
