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



#libs for dataset preperations

from sklearn.model_selection import train_test_split



#data visualization libs

import matplotlib.pyplot as plt

import seaborn as sns



#libs for ML algos

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier



#libs for result analysis

from sklearn import metrics





data = pd.read_csv("../input/Iris.csv") # Read the dataset from csv file
#dispaly first five rows from dataset



data.head()
#checking the dataset for null values



data.info()
#remove unwanted feature/Column

data.drop('Id', axis = 1, inplace=True)

data.head(3)
#Some insights from dataset



data.describe()
#devide dataset into train and test dataset by using 70:30 ratio

train, test = train_test_split(data, test_size = 0.3)
#verify train dataset

train.head(3)
#verify test dataset

test.head(3)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y=train.Species



test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_y =test.Species 
fig = data[data.Species == 'Iris-setosa'].plot(kind='scatter',x ='SepalLengthCm', y = 'SepalWidthCm', color = 'Red', label = 'Setosa' )



fig = data[data.Species == 'Iris-versicolor'].plot(kind='scatter',x ='SepalLengthCm', y = 'SepalWidthCm', color = 'Green', label = 'versicolor', ax = fig) 



fig = data[data.Species == 'Iris-virginica'].plot(kind='scatter',x ='SepalLengthCm', y = 'SepalWidthCm', color = 'Blue', label = 'virginica', ax = fig )



fig.set_xlabel("Sepal Length in CM")

fig.set_ylabel("Sepal Width in CM")

fig.set_title("Sepal Length VS Width")



fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = data[data.Species == 'Iris-setosa'].plot(kind='scatter',x ='PetalLengthCm', y = 'PetalWidthCm', color = 'Red', label = 'Setosa' )



fig = data[data.Species == 'Iris-versicolor'].plot(kind='scatter',x ='PetalLengthCm', y = 'PetalWidthCm', color = 'Green', label = 'versicolor', ax = fig) 



fig = data[data.Species == 'Iris-virginica'].plot(kind='scatter',x ='PetalLengthCm', y = 'PetalWidthCm', color = 'Blue', label = 'virginica', ax = fig )



fig.set_xlabel("Petal Length in CM")

fig.set_ylabel("Petal Width in CM")

fig.set_title("Petal Length VS Width")



fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
data.hist(edgecolor='black')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=data)



plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=data)



plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=data)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features

train_y=train.Species# output of our training data

test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features

test_y =test.Species 

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=data)



plt.show()
plt.figure(figsize=(7,4)) 

sns.heatmap(data.corr(),annot=True,cmap='cubehelix_r')

plt.show()
model = LogisticRegression()

model.fit(train_X, train_y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))
model = svm.SVC() 

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))
model=DecisionTreeClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))
model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))