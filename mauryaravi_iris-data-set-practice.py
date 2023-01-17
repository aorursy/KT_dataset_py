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
import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")
iris.head(5)
iris.info()
iris.drop('Id',axis=1,inplace=True)
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepals Length")
fig.set_ylabel("Sepals Width")
fig.set_title("Sepals Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='red', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()
train, test = train_test_split(iris, test_size = 0.3)
print(train.shape)
print(test.shape)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y =test.Species
model = svm.SVC(gamma='auto')
model.fit(train_X,train_y)
pred = model.predict(test_X)
print('The accuracy of the SVM is:',metrics.accuracy_score(pred,test_y))
model = LogisticRegression()
model.fit(train_X,train_y)
pred = model.predict(test_X)
print('The accuracy of the LR is:',metrics.accuracy_score(pred,test_y))
model = DecisionTreeClassifier()
model.fit(train_X,train_y)
pred = model.predict(test_X)
print('The accuracy of the DTC is:',metrics.accuracy_score(pred,test_y))
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_X,train_y)
pred = model.predict(test_X)
print('The accuracy of the LR is:',metrics.accuracy_score(pred,test_y))
a_index=list(range(1,11))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_y)
    pred=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(pred,test_y)))
plt.plot(a_index, a)
plt.xticks(x)