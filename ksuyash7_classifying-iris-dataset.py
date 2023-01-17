# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")
iris.head()
iris.info()
iris = iris.drop("Id", axis = 1)
iris
iris.columns
iris['Species'].unique()
iris = iris.dropna()
iris = iris[(iris['Species']!= 'undefined')]
iris[1,'Species'] = 'undefined'
iris = iris.drop((1, 'Species'), axis = 1)
iris.groupby(by = ['Species']).mean()
iris
fig = iris[iris['Species'] == 'Iris-virginica'].plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'green', label = 'virginica')

iris[iris['Species'] == 'Iris-setosa'].plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'orange', ax = fig, label = 'setosa')

iris[iris['Species'] == 'Iris-versicolor'].plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'blue', ax = fig, label = 'versicolor')

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = iris[iris['Species'] == 'Iris-setosa'].plot(kind = 'scatter',x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'orange', label = 'setosa')

iris[iris['Species'] == 'Iris-versicolor'].plot(kind = 'scatter',x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'blue', label = 'versicolor', ax = fig)

iris[iris['Species'] == 'Iris-virginica'].plot(kind = 'scatter',x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'green', label = 'virginica', ax = fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title("Sepal Length vs Width")

fig = plt.gcf()

fig.set_size_inches(10,6)
iris.info()
train, test = train_test_split(iris, test_size = 0.3)

print(train)

print(test)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_Y = train['Species']
test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_Y = test['Species']
train_X.head(2)

test_X.head(2)
train_Y.head(2)
model = svm.SVC() 

model.fit(train_X,train_Y) 

prediction=model.predict(test_X) 

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_Y))
model = LogisticRegression()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))
model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_Y))
model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_Y))
petal = iris[['PetalLengthCm','PetalWidthCm','Species']]

sepal = iris[['SepalLengthCm','SepalWidthCm','Species']]
train_p, test_p = train_test_split(petal, test_size = 0.3)

train_x_p = train_p[['PetalWidthCm','PetalLengthCm']]

train_y_p = train_p['Species']

test_x_p = test_p[['PetalWidthCm','PetalLengthCm']]

test_y_p = test_p['Species']





train_s, test_s = train_test_split(sepal, test_size = 0.3)

train_x_s = train_s[['SepalLengthCm','SepalWidthCm']]

train_y_s = train_s['Species']

test_x_s = test_s[['SepalLengthCm','SepalWidthCm']]

test_y_s = test_s['Species']
model=svm.SVC()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model=svm.SVC()

model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_y_s))
model = LogisticRegression()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=DecisionTreeClassifier()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))