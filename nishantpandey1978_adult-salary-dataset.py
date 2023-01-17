#Dataset Used - Adult from UCI ML Library
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import csv as csv

import os
#Reading the data
dataset_columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours per week','native-country','salary category']
dataset=pd.read_csv("../input/adult-salary/adult_salary.csv",na_values='?',names=dataset_columns)
#Checking any missing values in the dataset
dataset.isnull().sum()
dataset['salary category'].value_counts()
#Segregate the features and target from the dataset and columns based on data types

X=dataset.iloc[:,0:14]

y=dataset.iloc[:,14]

num_columns=[0,2,4,10,11,12]

str_columns=[1,3,5,6,7,8,9,13]
#fill the missing values in the dataset through SimpleImputer

from sklearn.impute import SimpleImputer

si=SimpleImputer(strategy='most_frequent')

temp=X.iloc[:,[1,6,13]]

temp=si.fit_transform(temp)

X.iloc[:,[1,6,13]]=temp

X.isnull().sum()
#Scaling the numeric values in the dataset

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X.iloc[:,[0,2,4,10,11,12]]=ss.fit_transform(X.iloc[:,[0,2,4,10,11,12]])
#Labeling the string values into numeric values

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



for i in str_columns:

    X.iloc[:,[i]]=le.fit_transform(X.iloc[:,[i]])

y=le.fit_transform(y)
#Segregating the dataset into Train and Test datasets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Applying Logistic Regression on the dataset

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()



lr.fit(X_train,y_train)

lr.score(X_test,y_test)
#Applying KNN Classifier on the dataset

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=7)



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
#Applying SVM Classifier on the dataset

from sklearn.svm import SVC

svm=SVC()



svm.fit(X_train,y_train)

svm.score(X_test,y_test)
#Applying Gaussian NB Classifier on the dataset

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()



nb.fit(X_train,y_train)

nb.score(X_test,y_test)
#Applying Decision Tree Classifier on the dataset

from sklearn.tree import DecisionTreeClassifier

dtf=DecisionTreeClassifier()



dtf.fit(X_train,y_train)

dtf.score(X_test,y_test)