#Import the packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
#Import the data

data = pd.read_csv("../input/Salary_Data.csv")
#First 5 lines of the data

data.head()
#Basic statistics of the data

data.describe()
#Basic info about the data

data.info()
#Correlation of the fields in the data

data.corr()
#Plot the relationship between the variables using pairplot

sns.pairplot(data)
#Separate Feature and Traget matrixs

x = data.iloc[:,:-1].values

y = data.iloc[:,1].values
#Split the train and test dataset

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
#Define the Machine Learning Alorithm

ml = LinearRegression()

#ml = LogisticRegression()
#Train the Machine Learning Algorithm (Learning)

ml.fit(x_train,y_train)
#Test the Machine Learning Algorithm (Prediction)

y_pred = ml.predict(x_test)
plt.scatter(x_test,y_test,color= 'red')

plt.plot(x_test,y_pred,color='blue')

plt.xlabel("Years of Experience")

plt.ylabel("Salary")
ml.score(x_test,y_test)