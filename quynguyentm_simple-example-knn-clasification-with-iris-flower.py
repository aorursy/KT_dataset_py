#Import required package for KNN model
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, neighbors 
from sklearn.model_selection import train_test_split #For split data to train and test set
from sklearn.metrics import accuracy_score #Calculate the accu of model
#Load data
np.random.seed(5) #Create the same value in every time we run code. Value is 32bits number.
data=datasets.load_iris()
#Set X, Y
Y=data.target
X=data.data
#Split data
X_train, X_test,Y_train, Y_test=train_test_split(X,Y,test_size=130)
X_test.shape[0]
#Model
#Using 1 nearest data point and P=2 is L2 norm. With N_neighbors = 1. we will using the 1 nearest data point,
#it will lead to Overfitting. We need to increase this values to fix it.
#Weights is set priority for training data points, which nearer test data point will higher weigh in predict
model=neighbors.KNeighborsClassifier(n_neighbors = 3, p = 2,weights='distance')

#Training model
model.fit(X_train, Y_train)
#After training, we using for predict Y with X_test and then cal accuracy of model
Y_predict=model.predict(X_test)
#The pecent rate of exactly classification with test data
accuracy_score(Y_test,Y_predict)

