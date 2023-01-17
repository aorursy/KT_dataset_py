#import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import library from guthub, changes names and load dataframe as 'iris'
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pd.read_csv(url, names=names)
iris
#Quick overview of data types and null values
iris.info()
iris.isnull().sum()
#scatter plot petal length v petal width by class
fig = iris[iris['class']=='Iris-setosa'].plot(kind='scatter',x='petal-length',y='petal-width',color='orange', label='Setosa')
iris[iris['class']=='Iris-versicolor'].plot(kind='scatter',x='petal-length',y='petal-width',color='blue', label='versicolor',ax=fig)
iris[iris['class']=='Iris-virginica'].plot(kind='scatter',x='petal-length',y='petal-width',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
#scatter plot sepal length v sepal width by class
fig = iris[iris['class']=='Iris-setosa'].plot(kind='scatter',x='sepal-length',y='sepal-width',color='orange', label='Setosa')
iris[iris['class']=='Iris-versicolor'].plot(kind='scatter',x='sepal-length',y='sepal-width',color='blue', label='versicolor',ax=fig)
iris[iris['class']=='Iris-virginica'].plot(kind='scatter',x='sepal-length',y='sepal-width',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
#Note: petal length/width has a tighter grouping than sepal length/width 
#import ML libraries
from sklearn.model_selection import train_test_split #train_test_split to split training and testing data
from sklearn import metrics #for model scoring
from sklearn import svm #svmmodel
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
#Use all features except Id for X, y = target variable
X = iris[['sepal-length','sepal-width','petal-length','petal-width']]
y = iris['class']
#Train 70% of data, test on 30% of data
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.7, random_state=1)
#SVM model
svm = svm.SVC()
svm.fit(X_train,y_train)

svm_outcomes = svm.predict(X_test)
print('The accuracy of the SVM is:',metrics.accuracy_score(svm_outcomes,y_test))
#kn neighbors model
kn = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
kn.fit(X_train,y_train)
kn_outcomes = kn.predict(X_test)
print('The accuracy of the KNN is',metrics.accuracy_score(kn_outcomes,y_test))
#decision tree model
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_outcome = dt.predict(X_test)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_outcome,y_test))