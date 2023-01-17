# Data in a csv format, each observation has a 4 measurements/features 
# and the species type.
from IPython.display import IFrame
IFrame('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', width=400, height=200)
#Import the load_iris function from datsets module
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Create bunch object containing iris dataset and its attributes.
iris = load_iris()
type(iris)
#Print the iris data
iris.data
#Names of 4 features (column names)
print(iris.feature_names)
#Integers representing the species: 0 = setosa, 1=versicolor, 2=virginica
print(iris.target)
# 3 classes of target
print(iris.target_names)
print(type(iris.data))
print(type(iris.target))
# we have a total of 150 observations and 4 features
print(iris.data.shape)
# Feature matrix in a object named X
X = iris.data
# response vector in a object named y
y = iris.target
print(X.shape)
print(y.shape)
# splitting the data into training and test sets (80:20)
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
#shape of train and test objects
print(X_train.shape)
print(X_test.shape)
# shape of new y objects
print(y_train.shape)
print(y_test.shape)
#import the KNeighborsClassifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier

#import metrics model to check the accuracy 
from sklearn import metrics
#Try running from k=1 through 25 and record testing accuracy
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
#Testing accuracy for each value of K
scores

%matplotlib inline
import matplotlib.pyplot as plt

#plot the relationship between K and the testing accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc
from sklearn.metrics import classification_report


print(classification_report(y_test,y_pred))