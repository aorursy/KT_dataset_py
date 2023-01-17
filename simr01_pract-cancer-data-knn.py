# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
%matplotlib inline

cancer = load_breast_cancer()  #https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
print(cancer.DESCR)

print(cancer.feature_names)
print(cancer.target_names)
type(cancer.data)
cancer.data.shape # rows and columns
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=cancer.target, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))
# To find the best number of neighbor (n_neighbors )value, as by default is 5

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state= 66)

training_accuracy=[]
test_accuracy=[]

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    # creat train and test cnn classifier
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the test set')
    
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend() 
knn_optimized = KNeighborsClassifier(6)
knn_optimized.fit(X_train, y_train)
print('Accuracy of KNN n-6, on the training set: {:.3f}'.format(knn_optimized.score(X_train, y_train)))
print('Accuracy of KNN n-6, on the test set: {:.3f}'.format(knn_optimized.score(X_test, y_test)))

# 10-fold cross-validation  - cv=10 , and setting Knn range till (1, 31)
from sklearn.model_selection import cross_val_score
K_range=list(range(1,31)) 
k_scores=[]

for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring = 'accuracy')
    k_scores.append(scores.mean())
    
print(k_scores)
import matplotlib.pyplot as plt
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(K_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
# 10-fold cross-validation with the best KNN model# 10-fol 
knn = KNeighborsClassifier(n_neighbors=20)
#print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

knn_optimized_cv = KNeighborsClassifier(6)
knn_optimized_cv.fit(X_train, y_train)
print('Accuracy of KNN n-20, on the training set: {:.3f}'.format(knn_optimized_cv.score(X_train, y_train)))
print('Accuracy of KNN n-20, on the test set: {:.3f}'.format(knn_optimized_cv.score(X_test, y_test)))
