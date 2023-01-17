# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import matplotlib.pyplot as plt

# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load iris data
data = pd.read_csv("../input/Iris.csv") #load the dataset
print(data.info()) #checking if there is any inconsistency in the dataset
# drop useless column
# data.drop('Id', axis=1, inplace=True)
print(data.head())
species = data.Species
speciesOfIris = species.unique()
print(speciesOfIris)
# split the data to train and test data
train, test = train_test_split(data, test_size=0.3)
#train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
train_X = train.drop('Species', axis=1)
train_y = train.Species
#test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
test_X = test.drop('Species', axis=1)
test_y = test.Species # output value of test data

print(train_X.head(2))
print(test_X.head(2))
#applying algorithm
# model = LogisticRegression()
# model.fit(train_X,train_y)
# prediction=model.predict(test_X)
# print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))
model = svm.SVC() #select the algorithm
model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm.
submission = pd.DataFrame({'Id': test.Id, 'Species': prediction})
submission.to_csv('submission.csv', index=False)