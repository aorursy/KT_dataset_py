import pandas as pd

import numpy as np

import scipy

import matplotlib.pyplot as plt

from sklearn import linear_model as ln

import os

print(os.listdir("../input"))
# Declaring an array

arr = np.array([[1,2,3],[4,5,6]])



print("Array dimensions:\n", arr.shape)

print("Array previous:\n", arr)
# Function to generate a Matrix with all values as 1:

identityMatrix = np.ones((2,2))

print("Identity Matrix:\n", identityMatrix)



#Function to stack so as to make a single Matrix horizontally:

x = np.hstack((identityMatrix,arr))

print("Stacking Arrays:\n", x)
# Dot Product

a = np.array([[7,8],[9,10]])

b = np.array([[11,12],[13,14]])

print(np.dot(a,b))
# Transpose

mat = np.array([[7,8],[9,10],[11,12],[13,14]])



print("Original Matrix:\n", mat)

print("Transpose Matrix:\n", np.transpose(mat))
# Function to calculate the inverse of a matrix

mat = np.array([[7,8],[9,10]])

print("Matrix Inverse:\n", np.linalg.inv(mat))
y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])

z = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])
# Generating regression coefficients

id = np.ones((8,1))

x = np.hstack((id,z))

beta = (np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y))

print(beta)
#Result - Calculation

yp1 = beta[0] + beta[1]*z

print(np.hstack((z,y,yp1)))
# Input DataFrame

d = pd.DataFrame(np.hstack((z,y)))

d.columns = ["x1","y"]

print(d)
# Linear Regression - model fitting

model = ln.LinearRegression()

results = model.fit(z,y)

print(model.intercept_, model.coef_)
# Result: Scikit - Learn

yp2 = model.predict(z)

print(yp2)
# Linear Regression representation using scatter plot

plt.title("Scatter Plot Representation",fontsize=16)

plt.scatter(z,y)

plt.plot(z,yp2, color="red")

plt.show()
# Prediction for new values

x1new = pd.DataFrame(np.hstack(np.array([[1],[0],[-0.12],[0.52]])))

x1new.columns = ["x1"]

yp2new = model.predict(x1new)

print(yp2new)
# Input DataFrame

y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])

x1 = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])

x2 = np.array([[1],[0],[1],[1],[0],[1],[0],[1]])
id = np.ones((8,1))

x = np.hstack((id, x1, x2))

print(x)
# Calculating Regression Coefficients

beta = (np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y))

print(beta)
# Result - Calculation

yp1 = beta[0] + beta[1]*x1 + beta[2]*x2

print(np.hstack((x,y,yp1)))
# Input DataFrame

d = pd.DataFrame(np.hstack((x1,x2,y)))

d.columns = ["x1","x2","y"]

print(d)
# Multiple Linear Regression - Model Fitting

inputDF = d[["x1","x2"]]

model = ln.LinearRegression()

result = model.fit(inputDF, y)



print(model.intercept_, model.coef_)
# Result: Scikit - Learn

yp2 = model.predict(inputDF)

yp2
# Prediction for new values

x1new = pd.DataFrame(np.hstack((np.array([[1],[0],[-0.12],[0.52]]),np.array([[1],[-1],[2],[0.7]]))))

x1new.columns = ["x1","x2"]

yp2new = model.predict(x1new)

print(np.hstack((x1new,yp2new)))
df = pd.read_csv("../input/survey.csv")

#df.head()

#df = df.rename(index=str,columns = ("Wr.Hnd":"WrHnd"))

df = df[['Wr.Hnd', 'Height']]

print(df.head())

print("------------------------------")

print(df.isnull().values.any())

print(df.isnull().sum())
#Checking for Null/Nan Values

df = df.dropna()

print("Check for NaN/Null values:\n", df.isnull().values.any())

print("Number of NaN/Null values:\n", df.isnull().sum())
# Simple Linear Regression

inputDF = df[["Wr.Hnd"]]

outcomeDF = df[["Height"]]

model = ln.LinearRegression()

results = model.fit(inputDF, outcomeDF)



print(model.intercept_, model.coef_)
df = pd.read_csv("../input/clock.csv")

print(df.head())

print("------------------------------------------------------------------------------")

print("Check for NaN/Null values:\n", df.isnull().values.any())

print("Number of NaN/Null values:\n", df.isnull().sum())
# Multiple Linear Regression 

inputDF = df[["Bidders","Age"]]

outcomeDF = df[["Price"]]

model = ln.LinearRegression()

results = model.fit(inputDF, outcomeDF)



print(model.intercept_, model.coef_)