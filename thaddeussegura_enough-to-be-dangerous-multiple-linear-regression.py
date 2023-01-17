#Numpy is used so that we can deal with array's, which are necessary for any linear algebra

# that takes place "under-the-hood" for any of these algorithms.



import numpy as np





#Pandas is used so that we can create dataframes, which is particularly useful when

# reading or writing from a CSV.



import pandas as pd





#Matplotlib is used to generate graphs in just a few lines of code.



import matplotlib.pyplot as plt



#Sklearn is a very common library that allows you to implement most basic ML algorithms.

#LabelEncoder, OneHotEncoder, and ColumnTransfomer are necessary since we have a field of categorical data.



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer



#Train_test_split will allow us to quickly split our dataset into a training set and a test set.



from sklearn.model_selection import train_test_split





#LinearRegression is the class of the algorithm we will be using.



from sklearn.linear_model import LinearRegression





#This will allow us to evaluate our fit using the R^2 score. 



from sklearn.metrics import r2_score



#read dataset from csv

dataset = pd.read_csv('../input/50-startups/50_Startups.csv')



#set independent variable using all rows, and all columns except for the last one.

X = dataset.iloc[:, :-1].values



#set the dependent variable using all rows, but ony the last column.

y = dataset.iloc[:, 4].values



#Lets look at our data

dataset

#create an object of the class LabelEncoder

labelencoder = LabelEncoder()



# Country column

ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')

X = ct.fit_transform(X)



#We need to omit one of the columns to avoid the dummy variable trap.

X = X[:, 1:]



#take a look at X now.

X
#This will create x and y variables for training and test sets.

#Here we are using 25% of our examples for the test set.



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#this sets the object regressor to the class of LinearRegression from the Sklearn library.

regressor = LinearRegression()



#this fits the model to our training data.

regressor.fit(X_train, y_train)
#Predict on our test set.

y_pred = regressor.predict(X_test)
#calculate the R^2 score

score = r2_score(y_test, y_pred)



#print out our score properly formatted as a percent.

print("R^2 score:", "{:.0%}".format(score))
#Prediction for a business in CA, with R&D of 160,000, Admin of 130,000 and Marketing of 300,000.

print(regressor.predict([[1, 0, 160000, 130000, 300000]]))