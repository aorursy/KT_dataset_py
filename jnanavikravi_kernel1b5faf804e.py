# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

#import load_iris function from datasets module

from sklearn.datasets import load_iris

#loading the dataset

iris = load_iris()

iris

#assign x 

x=iris.data

x
#assign y

y=iris.target

y
#to split the dataset into test and train data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
#check the shapes of all the train and test datas

x_train.shape

y_train.shape

x_test.shape

y_test.shape
#assign k values through a loop

from sklearn import metrics

k_range=range(5,15)



scores=[]



for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(x,y)

    pred=knn.predict(x)

#chcek accuracy

    scores.append(metrics.accuracy_score(y,pred))

    

scores
#assign the species

classes={0:'setosa',1:'versicolor',2:'virginica'}

#10 has the best score while checking the accuracy

knn=KNeighborsClassifier(n_neighbors=10)

knn.fit(x,y)





newx=[[3,4,5,2],[5,4,2,2]]

a=knn.predict(newx)

print(classes[a[0]])

print(classes[a[1]])