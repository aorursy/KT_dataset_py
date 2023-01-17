# This is Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:26:27 2017

@author: Arshid
"""
# Importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the training and test set
training=pd.read_csv('../input/train.csv')
testing=pd.read_csv('../input/test.csv')

training.drop(training.index[[213]], inplace=True)

# Importing thhe Imputer class used to impute the missing values
from sklearn.preprocessing import Imputer
imputer1=Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer1=imputer1.fit(training)
training=imputer1.transform(training)

# Train Test and Split
x_train=training[:,:-1]
x_test=testing.iloc[:,:-1].values
y_train=training[:,1]
y_test=testing.iloc[:,1].values

#Training the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# PLotting the training set
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('(Trainig set)')
plt.xlabel('X_Plane')
plt.ylabel('Y_Plane')
plt.show()

# Plotting the test set

plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('(Test set)')
plt.xlabel('X_Plane')
plt.ylabel('Y_Plane')
plt.show()


# Any results you write to the current directory are saved as output.