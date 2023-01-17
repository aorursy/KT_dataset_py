# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv") #Loading the datasets
iris.head(3) #the first three entries
iris.info() #checking for incosistencies in the data
iris.drop('Id',axis=1,inplace=True) #inplace = True because we are on the same copy
iris.head(3)#checking again
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='Red', label='Setosa')

fig = iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm',y='SepalWidthCm', color='Blue', label='Versicolor', ax=fig)

fig = iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='Green', label='Virginica', ax=fig)
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='Red', label='Setosa')

fig = iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='Blue', label='Versicolor',ax=fig)

fig = iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='Green', label='Virginica',ax=fig)

iris.hist(edgecolor='black',linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species', y='PetalLengthCm', data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species', y='PetalWidthCm', data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species', y='SepalLengthCm', data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species', y='SepalWidthCm', data=iris)
#importing packages for respective algorithms

from sklearn.linear_model import LogisticRegression #for logistic regression

from sklearn.neighbors import KNeighborsClassifier #for KNN

from sklearn import svm #for support vector machine

from sklearn import metrics #to test the accuracy

from sklearn.tree import DecisionTreeClassifier #for decision tree

from sklearn.model_selection import train_test_split #for splitting out data
iris.shape #to get the shape


iris.info() #to check for inconsistent data
plt.figure(figsize=(7,4))

sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r')

plt.show()

#Here it can be observed that Slength and width are not correlated while Plength and width are highly correlated

train, test= train_test_split(iris, test_size = 0.3)#to split the data for test and train

print(train.shape)

print(test.shape)#to check if bost the data is apt.
train_X=train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]#training the data

train_Y=train.Species

test_X=test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

test_Y=test.Species
test.shape
train.shape
train_X.head(2)
test_X.head(2)
train_Y.head()
#using support vector machines

model = svm.SVC()

model.fit(train_X,train_Y)#passing the input and output of the training model

prediction = model.predict(test_X)#predicting

print('The accuracy of svm is', metrics.accuracy_score(prediction,test_Y))#calculating the accuracy of the model by comparing it with test set
#using Logistic Regression

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of Logistic Regression is', metrics.accuracy_score(prediction, test_Y))
#now training petals and sepals seperately

petal=iris[['PetalLengthCm','PetalWidthCm','Species']]

sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]

train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)  #petals

train_X_p=train_p[['PetalWidthCm','PetalLengthCm']]

train_Y_p=train_p.Species

test_X_p=test_p[['PetalWidthCm','PetalLengthCm']]

test_Y_p=test_p.Species





train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal

train_X_s=train_s[['SepalWidthCm','SepalLengthCm']]

train_Y_s=train_s.Species

test_X_s=test_s[['SepalWidthCm','SepalLengthCm']]

test_Y_s=test_s.Species

#using support vector machine

model=svm.SVC()

model.fit(train_X_p,train_Y_p) 

prediction=model.predict(test_X_p) 

print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_Y_p))



model=svm.SVC()

model.fit(train_X_s,train_Y_s) 

prediction=model.predict(test_X_s) 

print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_Y_s))
#using logistic regression

model = LogisticRegression()

model.fit(train_X_p,train_Y_p) 

prediction=model.predict(test_X_p) 

print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_Y_p))



model.fit(train_X_s,train_Y_s) 

prediction=model.predict(test_X_s) 

print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_Y_s))
#thanks!