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
data = pd.read_csv("../input/heart.csv")
data.head()
data.shape
train = data.drop('target',axis = 1) 

print(train.head())
# Import train_test_split function

from sklearn.model_selection import train_test_split



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(train, data.target, test_size=0.3,random_state=109) # 70% training and 30% test
#Import svm model

from sklearn import svm



#Create a svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?

print("Precision:",metrics.precision_score(y_test, y_pred))



# Model Recall: what percentage of positive tuples are labelled as such?

print("Recall:",metrics.recall_score(y_test, y_pred))