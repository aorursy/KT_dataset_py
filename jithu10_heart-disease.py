# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt



#for model creation

from sklearn.model_selection import train_test_split



from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



#for decision tree classifcation

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



#for exporting

import graphviz

from graphviz import Source



#for randomforest classification

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification



#for knn classification

from sklearn.neighbors import KNeighborsClassifier



#for svm classification

from sklearn.svm import SVC

from sklearn import svm



#for mlp classification

from sklearn.neural_network import MLPClassifier







data = pd.read_csv("../input/heart.csv")

data.head()

#Checking balance factor

print(data["Presence"].value_counts())
#Splitting dataset into training and testing class

array = data.values

X = array[:,0:13]

Y = array[:,13]



X_train,X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="entropy",max_depth=3)



clf = clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_validation)



cm = confusion_matrix(Y_pred, Y_validation)



print("Accuracy of Decision tree classification:",metrics.accuracy_score(Y_validation, Y_pred))

print(cm)
plt.figure(figsize=(10,10))

tree.plot_tree(clf)
#random forest classification

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf = clf.fit(X_train,Y_train)



Y_pred = clf.predict(X_validation)

cm = confusion_matrix(Y_pred, Y_validation)



print("Accuracy random froest classification:",metrics.accuracy_score(Y_validation, Y_pred))

print(cm)
#knn clasification

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(X_train,Y_train)



Y_pred = knn.predict(X_validation)

cm = confusion_matrix(Y_pred, Y_validation)



print("Accuracy of knn classification:",metrics.accuracy_score(Y_validation, Y_pred))

print(cm)
#support vector machine Classifica

clf = svm.SVC(kernel='linear') 



clf.fit(X_train, Y_train)





Y_pred = clf.predict(X_validation)

cm = confusion_matrix(Y_pred, Y_validation)





print("Accuracy of SVM Classifier : ",metrics.accuracy_score(Y_validation, Y_pred))

print(cm)
#MultilayerPercetron classification



clf = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000,activation = 'relu',solver='adam',random_state=1)

clf=clf.fit(X_train, Y_train)



cm = confusion_matrix(Y_pred, Y_validation)

Y_pred = clf.predict(X_validation)



print("Accuracy of MLPClassifier : ",metrics.accuracy_score(Y_validation, Y_pred))

print(cm)