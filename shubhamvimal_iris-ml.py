# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')

iris.head()
iris.drop('Id',axis=1,inplace=True)

iris.head()
iris.info()
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red',label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue',label='Versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green',label='Virginica',ax=fig)

fig.set_xlabel('Sepal Length')

fig.set_ylabel('Sepal Width')

fig.set_title('Sepal Length vs Sepal Width')

fig = plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red',label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue',label='Versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green',label='Virginica',ax=fig)

fig.set_xlabel('Petal Length')

fig.set_ylabel('Petal Width')

fig.set_title('Petal Length vs Petal Width')

fig = plt.gcf()

fig.set_size_inches(10,6)

plt.show()
# importing alll the necessary packages to use the various classification algorithms



from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
plt.figure(figsize=(8,4)) 

ax = sns.heatmap(iris.corr(),annot=True,cmap='inferno') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

ax.set_ylim(4.0,0.0)

plt.yticks(rotation='horizontal')

plt.show()
train, test = train_test_split(iris, test_size = 0.3)# in this our main data is split into train and test

# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%

print(train.shape)

print(test.shape)
x_train = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features

y_train=train.Species# output of our training data



x_test= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features

y_test =test.Species   #output value of test data
# SVM 

model = svm.SVC()

model.fit(x_train,y_train)



pred = model.predict(x_test)

print('The accuracy of the SVM is:',metrics.accuracy_score(pred,y_test))
# Logistic Regression

model = LogisticRegression()

model.fit(x_train,y_train)



pred = model.predict(x_test)

print('The accuracy of Logistic Regression is:',metrics.accuracy_score(pred,y_test))
# Decision Tree

model = DecisionTreeClassifier()

model.fit(x_train,y_train)



pred = model.predict(x_test)

print('The acccuracy of DecisionTreeClassifier is :',metrics.accuracy_score(pred,y_test))
# Let's check the accuracy for various values of n for K-Nearest nerighbour

for i in range(1,11):

    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(x_train,y_train)

    pred = model.predict(x_test)

    print('The accuracy of KNeighborsClassifier with {} neighbors is: {}'.format(i,metrics.accuracy_score(pred,y_test)))
# Creating Petals And Sepals Training Data 

petals = iris[['PetalLengthCm','PetalWidthCm','Species']]

sepals = iris[['SepalLengthCm','SepalWidthCm','Species']]
p_train,p_test=train_test_split(petals,test_size=0.3,random_state=0)  #petals

x_train_p = p_train[['PetalLengthCm','PetalWidthCm']]

y_train_p = p_train.Species

x_test_p = p_test[['PetalLengthCm','PetalWidthCm']]

y_test_p = p_test.Species



s_train,s_test=train_test_split(sepals,test_size=0.3,random_state=0)  #sepals

x_train_s = x_train[['SepalLengthCm','SepalWidthCm']]

y_train_s = s_train.Species

x_test_s = s_test[['SepalLengthCm','SepalWidthCm']]

y_test_s = s_test.Species
# SVM

model = svm.SVC()

model.fit(x_train_p,y_train_p)

pred = model.predict(x_test_p)

print('The accuracy of SVM using Petals is:',metrics.accuracy_score(pred,y_test_p))



model.fit(x_train_s,y_train_s)

pred = model.predict(x_test_s)

print('The accuracy of SVM using Sepals is:',metrics.accuracy_score(pred,y_test_s))
# Logistic Regression

model = LogisticRegression()

model.fit(x_train_p,y_train_p)

pred = model.predict(x_test_p)

print('The accuracy of LogisticRegression using Petals is:',metrics.accuracy_score(pred,y_test_p))



model.fit(x_train_s,y_train_s)

pred = model.predict(x_test_s)

print('The accuracy of LogisticRegression using Sepals is:',metrics.accuracy_score(pred,y_test_s))
# DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(x_train_p,y_train_p)

pred = model.predict(x_test_p)

print('The accuracy of DecisionTreeClassifier using Petals is:',metrics.accuracy_score(pred,y_test_p))



model.fit(x_train_s,y_train_s)

pred = model.predict(x_test_s)

print('The accuracy of DecisionTreeClassifier using Sepals is:',metrics.accuracy_score(pred,y_test_s))
# K-Nearest Neighbours

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train_p,y_train_p)

pred = model.predict(x_test_p)

print('The accuracy of KNeighborsClassifier using Petals is:',metrics.accuracy_score(pred,y_test_p))



model.fit(x_train_s,y_train_s)

pred = model.predict(x_test_s)

print('The accuracy of KNeighborsClassifier using Petals is:',metrics.accuracy_score(pred,y_test_s))