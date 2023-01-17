import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import linear_model as lm
#Declaring an array

arr = np.array([[1,2,3],[4,5,6]])



print("Array dimensions:\n", arr.shape)

print("Array preview:\n", arr)
#Function to generate a Matrix with all values as 1.

identityMatrix = np.ones((2,2))

print("Identity Matrix:\n",identityMatrix)



#Function to stack so as to make a single Matrix horizontally.

x = np.hstack((identityMatrix,arr))

print("Stacking arrays:\n",x)
#Dot Product 

#Calculation: [[7*11+8*13, 7*12+8*14],[9*11+10*13, 9*12+10*14]]

a = np.array([[7,8],[9,10]]) 

b = np.array([[11,12],[13,14]]) 

print(np.dot(a,b))
#Transpose

mat = np.array([[7,8],[9,10],[11,12],[13,14]]) 



print("Original Matrix:\n", mat)

print("Tranposed Matrix:\n", np.transpose(mat))
# Function to calculate the inverse of a matrix

mat = np.array([[7,8],[9,10]])

print("Matrix Inverse:\n", np.linalg.inv(mat))
# 'x1' functions as an independent variable and 'y' as a dependent variable 

y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])

x1 = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])
#Generating regression coefficients

id = np.ones((8,1))

x = np.hstack((id,x1))

beta=(np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y))

print(beta)
#Result - Calculation

yp1 = beta[0]+beta[1]*x1

print(np.hstack((x1,y,yp1)))
#Input Dataframe

d = pd.DataFrame(np.hstack((x1,y)))

d.columns = ["x1","y"]

print(d)
#Linear Regression - model fitting

model = lm.LinearRegression()

results = model.fit(x1,y)

print(model.intercept_, model.coef_)
#Result: Scikit-Learn

yp2 = model.predict(x1)

print(yp2)
#Linear Regression representation using scatter plot

plt.scatter(x1,y)

plt.plot(x1,yp2, color="blue")

plt.show()
#Prediction for new values

x1new = pd.DataFrame(np.hstack(np.array([[1],[0],[-0.12],[0.52]])))

x1new.columns=["x1"]

yp2new = model.predict(x1new)

print(yp2new)
# Input Dataframe

y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])

x1 = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])

x2 = np.array([[1],[0],[1],[1],[0],[1],[0],[1]])
id = np.ones((8,1))

x = np.hstack((id,x1,x2))

print(x)
# Calculating regression coefficients 

beta=(np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y))

print(beta)
#Result - Calculation

yp1 = beta[0]+beta[1]*x1+beta[2]*x2

print(np.hstack((x,y,yp1)))
#Input Dataframe

d = pd.DataFrame(np.hstack((x1,x2,y)))

d.columns = ["x1","x2","y"]

print(d)
#Multiple Linear Regression - Model Fitting

inputDF = d[["x1","x2"]]

model = lm.LinearRegression()

results = model.fit(inputDF,y)



print(model.intercept_, model.coef_)
#Result: Scikit-Learn

yp2 = model.predict(inputDF)

yp2
#Prediction for new values

x1new = pd.DataFrame(np.hstack((np.array([[1],[0],[-0.12],[0.52]]),np.array([[1],[-1],[2],[0.77]]))))

x1new.columns=["x1","x2"]

yp2new = model.predict(x1new)

print(np.hstack((x1new,yp2new)))
d=pd.read_csv("../input/survey.csv")

d=d.rename(index=str,columns={"Wr.Hnd":"WrHnd"})

d = d[["WrHnd","Height"]]

#.head()

print(d.isnull().values.any())

print(d.isnull().sum())
#Checking for Null/NaN values

d = d.dropna()

print("Check for NaN/null values:\n",d.isnull().values.any())

print("Number of NaN/null values:\n",d.isnull().sum())
# Simple Linear Regression 

inputDF = d[["WrHnd"]]

outcomeDF = d[["Height"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)



print(model.intercept_, model.coef_)
d = pd.read_csv("../input/clock.csv")

print(d.head())

print("Check for NaN/null values:\n",d.isnull().values.any())

print("Number of NaN/null values:\n",d.isnull().sum())
#Multiple Linear Regression

inputDF = d[["Bidders","Age"]]

outputDF = d[["Price"]]



model = lm.LinearRegression()

results = model.fit(inputDF,outputDF)



print(model.intercept_, model.coef_)