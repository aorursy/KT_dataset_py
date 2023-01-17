import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



%matplotlib inline
from PIL import Image

jpgfile = Image.open("E:\Datasets\download.jfif")

 

print(jpgfile.bits, jpgfile.size, jpgfile.format)

jpgfile
jpgfile2 = Image.open("E:\Datasets\ml.png")

 

print(jpgfile.bits, jpgfile.size, jpgfile.format)

jpgfile2
iris.head()
iris.isnull().values.any()
iris.shape
iris.tail()
iris.info()
iris.describe()
iris['Species'].value_counts()
iris1 = iris.drop('Id',axis =1)
iris1.head()
g = sns.pairplot(iris1, hue='Species', markers='+')

g
fig = iris1[iris1.Species == 'Iris-setosa'].plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'orange',label = 'Setosa')

iris1[iris1.Species == 'Iris-versicolor'].plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'blue',label = 'Versicolor', ax = fig)

iris1[iris1.Species == 'Iris-virginica'].plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'green',label = 'Virginica', ax = fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
iris1.head()
iris1.hist(edgecolor='black')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
x=iris.drop(['Id','Species'],axis = 1)

y=iris['Species']
print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.3, random_state = 10) 
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
plt.figure(figsize=(8,8)) 

sns.heatmap(iris1.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
logr = LogisticRegression()

logr.fit(X_train,Y_train)

Y_pred = logr.predict(X_test)

acc_log = metrics.accuracy_score(Y_pred,Y_test)

print("The accuracy of the Logistic Regression is", acc_log)
dt = DecisionTreeClassifier()

dt.fit(X_train,Y_train)

y_pred = dt.predict(X_test)

acc_dt = metrics.accuracy_score(y_pred,Y_test)

print('The accuracy of the Decision Tree is', acc_dt)
sv = svm.SVC() # selecting the algorithm

sv.fit(X_train,Y_train)

Y_pred = sv.predict(X_test)

acc_sv = metrics.accuracy_score(Y_pred,Y_test)

print("the accuracy of the support vector machine is", acc_sv)

knn = KNeighborsClassifier()

knn.fit(X_train,Y_train)

y_pred = knn.predict(X_test)

acc_knn = metrics.accuracy_score(y_pred,Y_test)

print("The accuracy score of the k nearest neighbours algorithm is", acc_knn)
models = pd.DataFrame({'Model': ['Logistic Regression','Decision Tree','Support Vector Machine','K-Nearest Neighbours'], 

                       'Score' : [acc_log,acc_dt,acc_sv,acc_knn]})

models.sort_values(by = 'Score', ascending = False)