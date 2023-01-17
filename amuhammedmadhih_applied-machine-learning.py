import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns



iris = pd.read_csv("/kaggle/input/iris/Iris.csv")



iris.head()
iris.info() 
#Remove unneeded column

iris.drop('Id',axis=1,inplace=True)
#EDA With Iris

fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
iris.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
# importing alll the necessary packages to use the various classification algorithms

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
iris.shape 
plt.figure(figsize=(7,4)) 

sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()
train, test = train_test_split(iris, test_size = 0.3)



print(train.shape)

print(test.shape)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features

train_y=train.Species# output of our training data

test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features

test_y =test.Species   #output value of test data
train_X.head(2)
test_X.head(2)
train_y.head()  ##output of the training data
# Support Vector Machine



model = svm.SVC() 

model.fit(train_X,train_y) 

prediction=model.predict(test_X)

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))

# Logistic Regression



model = LogisticRegression()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))
# Decision Tree

model=DecisionTreeClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))
# KNN

model=KNeighborsClassifier(n_neighbors=3)

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))
a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))

plt.plot(a_index, a)

plt.xticks(x)
# Creating Petals And Sepals Training Data

petal=iris[['PetalLengthCm','PetalWidthCm','Species']]

sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]
train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)  #petals

train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]

train_y_p=train_p.Species

test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]

test_y_p=test_p.Species





train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal

train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]

train_y_s=train_s.Species

test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]

test_y_s=test_s.Species
# SVM

model=svm.SVC()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model=svm.SVC()

model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_y_s))
#Logistic Regression

model = LogisticRegression()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
# Decision Tree

model=DecisionTreeClassifier()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))

#KNN

model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
