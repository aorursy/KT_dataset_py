import numpy as np #Numpy for basic operation

import pandas as pd #Pandas for reading input

import matplotlib.pyplot as plt #For visualisations

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/Iris.csv");

data.head()
data.shape
data.describe()
data.info()
data.drop('Id',axis=1,inplace=True)

plt.figure(figsize=(15,8))

data.hist()

plt.show()
data.groupby('Species').size()
fig = data[data.Species=="Iris-setosa"].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color="green")

data[data.Species=="Iris-versicolor"].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color="blue",ax=fig)

data[data.Species=="Iris-virginica"].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color="orange",ax=fig)

plt.show()
fig = data[data.Species=="Iris-setosa"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color="green")

data[data.Species=="Iris-versicolor"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color="blue",ax=fig)

data[data.Species=="Iris-virginica"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color="orange",ax=fig)

plt.show()
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.cross_validation import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
train ,test = train_test_split(data, test_size=0.3)
train.shape
test.shape
x_train = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_train = train.Species

x_test = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_test = test.Species
x_train.head()
y_train.head()
x_test.head()
#Support Vector Machine

model = svm.SVC()

model.fit(x_train,y_train)

prediction = model.predict(x_test)

metrics.accuracy_score(prediction,y_test)

#Logistic Regression

model = LogisticRegression()

model.fit(x_train,y_train)

prediction = model.predict(x_test)

metrics.accuracy_score(prediction,y_test)
#Decision Tree

model=DecisionTreeClassifier()

model.fit(x_train,y_train)

prediction = model.predict(x_test)

metrics.accuracy_score(prediction,y_test)
#K-Nearest Neighbours

model=KNeighborsClassifier(n_neighbors=3)

model.fit(x_train,y_train)

prediction = model.predict(x_test)

metrics.accuracy_score(prediction,y_test)
#For different n neighbours

a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(x_train,y_train)

    prediction = model.predict(x_test)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))

plt.plot(a_index, a)

plt.xticks(x)