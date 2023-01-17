# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# coding: utf-8
# In[3]:

#import load_iris function from datasets module
from sklearn.datasets import load_iris


# In[2]:

from matplotlib import pyplot as plt

%matplotlib inline
#save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)
# printing the data and target

print("Feature names:")
print(iris.feature_names)
print("Data(X):")
print(iris.data[:3])
print("Target names:[0   1   2]")
print(iris.target_names)
print("Target(y):")
print(iris.target)
#Each row represents one flower and four columns represents four measurements
# Check the type and shape of features and target. 
print(type(iris.data))
print(type(iris.target))
print(iris.data.shape)
print(iris.target.shape)
# store feature matrix in "X" - captialized as it represents matrix
X = iris.data

# store response vector in "y" - capitalized as it represents a vector
y = iris.target

import pandas as pd
data = pd.DataFrame(X, columns=iris.feature_names)
data['target'] = pd.DataFrame(y.reshape(-1,1), columns=["target"])
data.head(5)
from matplotlib import cm

# Petal Length and Width
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121)
ax.set_title('Iris Classification Based on Sepal length and width', fontsize=20)
ax.set_xlabel('Sepal Length (cm)', fontsize=18)
ax.set_ylabel('Sepal Width (cm)',fontsize=18)
# setting grid lines 
ax.grid(True,linestyle='-',color='0.75')
ax.scatter(data['sepal length (cm)'], data['sepal width (cm)'], s = 100, 
           c=data['target'], marker='o', cmap=cm.jet)

# Sepal Length and Width
ax = fig.add_subplot(122)
ax.set_title("iris Classification Based on Petal length and width", fontsize=20)
ax.set_xlabel('Petal Length (cm)', fontsize=18)
ax.set_ylabel('Petal width (cm)', fontsize=18)
ax.grid(True, linestyle='-', color='0.75')
ax.scatter(data['petal length (cm)'], data['petal width (cm)'], s = 100, 
           c=data['target'], marker='s', cmap=cm.jet)


# import class
from sklearn.neighbors import KNeighborsClassifier

# instantiating estimator. Choosing K=1
knn = KNeighborsClassifier(n_neighbors=1)

# model building
knn.fit(X,y)
# predicting out-of-sample data
knn.predict([3,5,4,2])
# Multiple Observations of out of sample data
X_new = [[3,5,4,2],[5,4,3,2]]
knn.predict(X_new)
# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X,y)

#predict the response for new observations
knn.predict(X_new)
from sklearn import metrics

# Split X and y into training and testing sets. Here 40% is for testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 4)
# printing the X shapes
print(X_train.shape)
print(X_test.shape)

# printing the y shapes
print(y_train.shape)
print(y_test.shape)
# instantiate
knn = KNeighborsClassifier(n_neighbors = 5)

# train
knn.fit(X_train, y_train)

# predict
y_pred = knn.predict(X_test)

# test accuracy
print(metrics.accuracy_score(y_test, y_pred))
# instantiate class
knn = KNeighborsClassifier(n_neighbors = 1)

# train
knn.fit(X_train, y_train)

# predict
y_pred = knn.predict(X_test)

# test accuracy
print(metrics.accuracy_score(y_test, y_pred))
k_range = range(1, 31)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
# plot the relationship between K and testing accuracy
plt.plot(k_range, scores, color="green")
plt.xlabel("K-Value")
plt.ylabel("Accuracy Score") 
# - Choosing K between 10-15 based on the graph

# In[26]:

# instantiate class
knn = KNeighborsClassifier(n_neighbors=12)

# training the model using all data
knn.fit(X, y)

#making predictions on out-of-sample data
knn.predict([3, 5, 4, 2])
