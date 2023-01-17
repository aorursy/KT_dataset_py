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
#importing data from sklearn library
from sklearn.datasets import load_iris
#data loaded from library into dataframe
iris=load_iris()
#viewing data
iris
#data contains features data.
#features of this data are sepal length, sepal width, petal length, petal width. All of them are in cm.
#target names contain label name.
#target contains labels of different species; 0 for setos, 1 for versicolor, 2 for virginica
iris.data
iris.target_names
iris.target
#since this dataset does not include testing data, therefore we will split the train data in to training and testing data
#importing train test split method
from sklearn.model_selection import train_test_split
#saving data into different dataframes for further use after splitting
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
#x_train contains train data from iris.data, y_train contains lables from iris.target for training purpose
#x_test contains data for testing from iris.data,y_test contains lables fromiris.target for testing purpose
x_train
x_test
y_train
y_test
#since it is a classifier problem, therefore we will use Classification Algo.
#we will use Logistic Regression for prediction of species
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
#training model here
model.fit(x_train,y_train)
#now model is trained 
#checking accuracy of this trained model on train data
print('Accuracy',model.score(x_train,y_train))
#now lets check for its accuracy on train data
print('Accuracy',model.score(x_test,y_test))
#lets check accuracy on some other classification models
from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)
print('Accuracy',model2.score(x_train,y_train))
#lets check for its accuracy on test data.
print('Accuracy',model2.score(x_test,y_test))
#lets check accuracy using SVM model
from sklearn.svm import SVC
model3=SVC()
model3.fit(x_train,y_train)

print('Accuracy',model3.score(x_train,y_train))
print('Accuracy',model3.score(x_test,y_test))
#we get a better accuracy on test data using SVM model. 
