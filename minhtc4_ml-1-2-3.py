from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics.cluster import homogeneity_score

import numpy as np

import pandas as pd
iris = load_iris()
print("""





EXERCISE 01

Write a program to classify IRIS flowers by using the K-Means algorithm. Compare

the output when apply or not apply scaling and PCA techniques.





""")
km = KMeans(n_clusters=3,random_state=321)

km.fit(iris.data)

ypred = km.labels_

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(iris.target, ypred, target_names=target_names))
print (confusion_matrix(iris.target, ypred))

scaler = MaxAbsScaler()

Xtrain = scaler.fit_transform(iris.data)
from sklearn.decomposition import PCA

pca = PCA(0.95)

Xtf = pca.fit_transform(Xtrain)

Xtf[0:10]
km = KMeans(n_clusters=3,random_state=321)

km.fit(Xtf)

ypred = km.labels_

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(iris.target, ypred, target_names=target_names))
print (confusion_matrix(iris.target, ypred))
# EXERCISE 02

# Write a program to classify IRIS flowers by using the Hierarchical algorithm.

# Compare the output when apply or not apply scaling and PCA techniques.

from sklearn.cluster import AgglomerativeClustering
ward = AgglomerativeClustering(n_clusters=3).fit(iris.data)

y_pred = ward.labels_

y_pred
print(classification_report(iris.target, ypred, target_names=target_names))

print (confusion_matrix(iris.target, ypred))
# PCA and scaler

scaler = MaxAbsScaler()

Xtrain = scaler.fit_transform(iris.data)

pca = PCA(0.95)

Xtf = pca.fit_transform(Xtrain)

Xtf[0:10]
ward = AgglomerativeClustering(n_clusters=3,linkage='average').fit(Xtf)

y_pred = ward.labels_

y_pred
print(classification_report(iris.target, ypred, target_names=target_names))

print (confusion_matrix(iris.target, ypred))

print("Compute structured hierarchical clustering...")

from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(iris.data, n_neighbors=10, include_self=False)

connectivity
# PCA and scaler

scaler = MaxAbsScaler()

Xtrain = scaler.fit_transform(iris.data)

pca = PCA(0.95)

Xtf = pca.fit_transform(Xtrain)

ward = AgglomerativeClustering(n_clusters=3, connectivity=connectivity,

                               linkage='single').fit(Xtf)

ward.labels_
print(classification_report(iris.target, ypred, target_names=target_names))

print (confusion_matrix(iris.target, ypred))

# EXERCISE 03

# Write a program to classify and predict fishes in fish.csv. Compare the output

# among supervised and unsupervised algorithms and apply or not apply scaling and

# PCA techniques.
dt_fish = pd.read_csv('../input/dataset/fish.csv',names=['C1','C2','C3','C4','C5','C6','C7','C8'])

dt_fish.head()
#Classifier

X = dt_fish.iloc[:,1:-1]

y = dt_fish.iloc[:,-1]

from sklearn.model_selection import cross_val_score

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

clf = tree.DecisionTreeClassifier()

y_pred = cross_val_score(clf, X, y, cv=10)

print('Decision tree= ',y_pred.mean())

clf = svm.SVC(gamma='scale')

y_pred = cross_val_score(clf, X, y, cv=10)

print('SVM= ',y_pred.mean())

clf = GaussianNB()

y_pred = cross_val_score(clf, X, y, cv=10)

print('Navibayes= ',y_pred.mean())

knn = KNeighborsClassifier(n_neighbors=1)

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

print('KNN= ',scores.mean())

clf = LogisticRegression(solver='newton-cg', multi_class='multinomial')

scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

print('Logistic= ',scores.mean())
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.3,random_state=31)

clf = tree.DecisionTreeClassifier()

clf.fit(Xtrain,ytrain)

clf.score(Xtest,ytest)
from sklearn.cluster import KMeans

km = KMeans(4,random_state=19)

km.fit(X)

ypred = km.labels_

ypred
print(classification_report(y, ypred))

print(confusion_matrix(y, ypred))

# PCA and scaler

scaler = StandardScaler()

Xtrain = scaler.fit_transform(X)

pca = PCA(0.95)

Xtf = pca.fit_transform(Xtrain)

Xtf[0:10]
km = KMeans(4,random_state=42)

km.fit(Xtf)

ypred = km.labels_

ypred
print(classification_report(y, ypred))

print(confusion_matrix(y, ypred))

ward = AgglomerativeClustering(n_clusters=4).fit(X)

y_pred = ward.labels_

y_pred
print(classification_report(y, ypred))

print(confusion_matrix(y, ypred))

ward = AgglomerativeClustering(n_clusters=4).fit(Xtf)

y_pred = ward.labels_

y_pred
print(classification_report(y, ypred))

print(confusion_matrix(y, ypred))
