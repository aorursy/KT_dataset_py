# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris=pd.read_csv("../input/Iris.csv")
iris.head(5)

iris.info()

iris.Species.unique()
iris.drop('Id',axis=1,inplace=True)
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title("Petal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()

iris.hist(edgecolor='black',linewidth=1,grid=False,color='grey')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
from sklearn.model_selection import train_test_split

from sklearn import metrics
plt.figure(figsize=(7,4))

sns.heatmap(iris.corr(),annot=True)

plt.show()
y_iris=iris.Species

x_iris=iris.drop(columns='Species')

print(x_iris.head())

print(y_iris.head())
X_train, X_test, y_train, y_test = train_test_split(x_iris, y_iris, test_size=0.4, random_state=42)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train.head()
y_train.head()
#Logistic Regression

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)

predicted=model.predict(X_test)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (Logistic Regression) : ',accuracy)
#K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

model=KNeighborsClassifier(n_neighbors=3)

model.fit(X_train,y_train)

predicted=model.predict(X_test)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (KNN) : ',accuracy)

print(confusion_matrix(y_test,predicted))
#Support Vector Machines

from sklearn.svm import SVC

model=SVC(gamma='auto')

model.fit(X_train,y_train)

predicted=model.predict(X_test)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (SVC) : ',accuracy)
#Decission Tree

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(X_train,y_train)

predicted=model.predict(X_test)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (Decission Tree) : ',accuracy)
X_train2=X_train.drop(columns=['SepalLengthCm','PetalLengthCm'])

X_train2.head()
X_test2=X_test.drop(columns=['SepalLengthCm','PetalLengthCm'])

X_test2.head()
#Logistic Regression

model=LogisticRegression()

model.fit(X_train2,y_train)

predicted=model.predict(X_test2)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (Logistic Regression) : ',accuracy)
#K Nearest Neighbors

model=KNeighborsClassifier(n_neighbors=3)

model.fit(X_train2,y_train)

predicted=model.predict(X_test2)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (KNN) : ',accuracy)

print(confusion_matrix(y_test,predicted))
#Support Vector Machines

model=SVC(gamma='auto')

model.fit(X_train2,y_train)

predicted=model.predict(X_test2)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (SVC) : ',accuracy)

print(confusion_matrix(y_test,predicted))
#Decission Tree

model=DecisionTreeClassifier()

model.fit(X_train2,y_train)

predicted=model.predict(X_test2)

accuracy=metrics.accuracy_score(predicted,y_test)

print('Accuracy (Decission Tree) : ',accuracy)