# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

"""
    Created on Friday March 01
    @author : Tanveer Baba
"""
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
%matplotlib inline
# Input data files are available in the "../input/" directory.

train_data = pd.read_csv('../input/train.csv') #Traning Data
test_data = pd.read_csv('../input/test.csv') #Testing Data

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

print(train_data.head()) #Returns first 5 rows of Train_data
print(test_data.head())  #Returns first 5 rows of Test_data

#Returns Descriptive Statistics that summarizes the central Tendency
print(train_data.describe())
print(test_data.describe())

#Information of a DataFrames
print(train_data.info())
print(test_data.info())

#Prints the Shape of a DataFrames
print(test_data.shape)
print(train_data.shape)

#Droping of Missing Data
test_data = test_data.dropna()
train_data = train_data.dropna()

#Prints the Shape of a DataFrames after droping
print(test_data.shape)
print(train_data.shape)

#Visualizing Train_data and Test_data
sns.jointplot(x = 'x', y = 'y', data = train_data)
sns.jointplot(x = 'x', y = 'y', data = test_data)

#Creation of Linear Model Object
lm = LinearRegression()

#Slicing of Datasets
x_train = pd.DataFrame(train_data.iloc[:,0].values)
y_train = pd.DataFrame(train_data.iloc[:,1].values)

x_test = pd.DataFrame(test_data.iloc[:,0].values)
y_test = pd.DataFrame(test_data.iloc[:,1].values)

#Training the Model by training dataset
lm.fit(x_train,y_train)

#Prints the Accuracy of Model
accuracy = round(lm.score(x_train,y_train) *100,2)
print('Accuracy:', accuracy)

#Prints the Coefficients
print('Coefficients', lm.coef_)

#Estimated prediction of y_test values based on trained model
predictions = lm.predict(x_test)
#Visualizing the Training Dataset
plt.figure(figsize = (12,6))
plt.scatter(x_train,y_train)
plt.plot(x_train,lm.predict(x_train), color = 'red')
#plt.xlim(5)
#plt.ylim(2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Data')
#Visualizing the Test Dataset

plt.figure(figsize = (12,6))
plt.scatter(x_test,y_test)
plt.plot(x_train,lm.predict(x_train), color = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Data')
#plt.xlim(5)
#plt.ylim(2)
#Real Test Values Versus Predicted Test Values
plt.scatter(y_test,predictions)
plt.xlabel('Y Values')
plt.ylabel('Predicted Values')
plt.title('R_values VS P_values')
#Model was correct choice for data because of Normal distribution
sns.distplot((y_test-predictions))