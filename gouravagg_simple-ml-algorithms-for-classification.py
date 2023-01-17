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
#Extracting the data

dataset=pd.read_csv('../input/Iris.csv')
dataset.head()
#dropping 1st column i.e 'Id' as it is not required

dataset=dataset.drop('Id',axis=1)
#Data Visualization

plt.scatter(dataset[dataset.Species=='Iris-setosa'].SepalLengthCm,dataset[dataset.Species=='Iris-setosa'].SepalWidthCm,c='b',marker='o',s=10,label='Iris-setosa')

plt.scatter(dataset[dataset.Species=='Iris-virginica'].SepalLengthCm,dataset[dataset.Species=='Iris-versicolor'].SepalWidthCm,c='y',marker='o',s=10,label='Iris-versicolor')

plt.scatter(dataset[dataset.Species=='Iris-virginica'].SepalLengthCm,dataset[dataset.Species=='Iris-virginica'].SepalWidthCm,c='r',marker='o',s=10,label='Iris-virginica')

plt.legend()

plt.xlabel('SepalLengthCm')

plt.ylabel('SepalWidthCm')



plt.show()
plt.scatter(dataset[dataset.Species=='Iris-setosa'].PetalLengthCm,dataset[dataset.Species=='Iris-setosa'].PetalWidthCm,c='b',marker='o',s=10,label='Iris-setosa')

plt.scatter(dataset[dataset.Species=='Iris-virginica'].PetalLengthCm,dataset[dataset.Species=='Iris-versicolor'].PetalWidthCm,c='y',marker='o',s=10,label='Iris-versicolor')

plt.scatter(dataset[dataset.Species=='Iris-virginica'].PetalLengthCm,dataset[dataset.Species=='Iris-virginica'].PetalWidthCm,c='r',marker='o',s=10,label='Iris-virginica')

plt.legend()

plt.xlabel('PetalLengthCm')

plt.ylabel('PetalWidthCm')



plt.show()
dataset.hist(edgecolor='black')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=dataset)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=dataset)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=dataset)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=dataset)
#Checking any missing value

dataset.isnull().sum()
X=dataset.iloc[:,[0,1,2,3]].values

y=dataset.iloc[:,[4]].values
#Splitting data into train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

log_classifier=LogisticRegression()

log_classifier.fit(X_train,y_train)

y_log=log_classifier.predict(X_test)
#Support Vector Machines

from sklearn.svm import SVC

svm_classifier=SVC()

svm_classifier.fit(X_train,y_train)

y_svm=svm_classifier.predict(X_test)

#K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

knn_classifier=KNeighborsClassifier(n_neighbors=5)

knn_classifier.fit(X_train,y_train)

y_knn=knn_classifier.predict(X_test)
#Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier

dtc_classifier=DecisionTreeClassifier()

dtc_classifier.fit(X_train,y_train)

y_dtc=dtc_classifier.predict(X_test)
#Computing accuracy of each algorithm

from sklearn.metrics import confusion_matrix,accuracy_score

print('Accuracy of Logistic Regression:')

print(accuracy_score(y_test,y_log))

print('Accuracy of Support Vector Machine:')

print(accuracy_score(y_test,y_svm))

print('Accuracy of K Nearest Neighbors:')

print(accuracy_score(y_test,y_knn))

print('Accuracy of Decision Tree Classification:')

print(accuracy_score(y_test,y_dtc))
