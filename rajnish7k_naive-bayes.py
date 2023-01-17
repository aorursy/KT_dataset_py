# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



adult= pd.read_csv("../input/adult.csv")



# Any results you write to the current directory are saved as output.

#'fnlwgt' make no sense. So dropping it.

#'educationNum' and 'education' are equivalent. So dropping 'Education' also

adult.drop(['fnlwgt', 'education'], axis=1, inplace=True)



#fill missing value with most frequent value

adult['workclass'] = adult['workclass'].fillna(adult['workclass'].mode()[0])

adult['occupation'] = adult['occupation'].fillna(adult['occupation'].mode()[0])

adult['native.country'] = adult['native.country'].fillna(adult['native.country'].mode()[0])



# here 'income' is our target variable we need to seperate from our main dataset beofore transformation

# and also creating dummies for y

x_train = pd.get_dummies(adult)



y_train= adult.iloc[:,-1].map({ "<=50K": 0, ">50K": 1 })



adult=adult.drop(columns=["income"],axis=1)



#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets

model.fit(x_train,y_train)



#Predict Output

y_pred_train=model.predict(x_train)



#average accuracy



avg_accuracy_train=100*(y_pred_train==y_train).sum()/y_train.shape[0]



print("Average training accuracy is : ",avg_accuracy_train)



y_train_zero=100*(y_pred_train[y_train==0]==0).sum()/len(y_train==0)

y_train_one=100*(y_pred_train[y_train==1]==1).sum()/len(y_train==1)

print("Average Class 0 training accuracy is : ",y_train_zero)

print("Average Class 1 training accuracy is : ",y_train_one)