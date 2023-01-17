import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

iris = pd.read_csv("../input/Iris.csv") #load the dataset

iris.head()

iris["Species"].value_counts()
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
# importing alll the necessary packages to use the various classification algorithms

from sklearn.cross_validation import train_test_split #to split the dataset for training and testing

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

train, test = train_test_split(iris, test_size = 0.3)# in this our main data is split into train and test

# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%

print(train.shape)

print(test.shape)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features

train_y=train.Species# output of our training data

test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features

test_y =test.Species   #output value of test data
model = svm.SVC() #select the algorithm

model.fit(train_X,train_y) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 

#we pass the predicted output by the model and the actual output
petal=iris[['PetalLengthCm','PetalWidthCm','Species']]

sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]



train_p,test_p=train_test_split(petal,test_size=0.3)  #petals

train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]

train_y_p=train_p.Species

test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]

test_y_p=test_p.Species





train_s,test_s=train_test_split(sepal,test_size=0.3)  #Sepal

train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]

train_y_s=train_s.Species

test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]

test_y_s=test_s.Species
model=svm.SVC()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model=svm.SVC()

model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_y_s))