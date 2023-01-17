import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as lm

arr = np.array([[1,2,3],[4,5,6]])
print('Array dimension:\n', arr.shape)
print("Array preview:\n",arr)
#identity matrix is a matrix where all the values are 1. we will use numpy to create an identity matrix.
identityMatrix = np.ones((2,2))
print("Identity Matrix:\n", identityMatrix)

#to transpose the matrix we use numpy hstack function

x = np.hstack((identityMatrix,arr))
print("stacking arrays:\n",x)

#Dot Product
#multiply row of one matrix with column of another matrix
#Calculation: [[7*11+8*13, 7*12+8*14],[9*11+10*13, 9*12+10*14]]

a = np.array([[7,8],[9,10]])
b = np.array([[11,12],[13,14]])

print(np.dot(a,b))
#matrix transpose is function is to convert the matrix rows to column or from coulmn. to row
mat = np.array([[7,8],[9,10],[11,12],[13,14]])
print("original array:\n", mat)
print("transposed matrix:\n", np.transpose(mat))
#function to calculate inverse of a matrix
mat = np.array([[7,8],[9,10]])
print("Matrix Inverse:\n", np.linalg.inv(mat))
#simple linear regression using NumPy
y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])
x1 = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])

#Result - Calculation
yp1 = beta[0]+beta[1]*x1
print(np.hstack((x1,y,yp1)))
# we would be doing same linear regression but using scikit learn instead of pandas
d = pd.DataFrame(np.hstack((x1,y)))
d.columns =['x1','y']
print(d)
# now following the drill of using scikit learn for using pre build models step 1 declare the model, step 2 fir the mode.
model = lm.LinearRegression()
results = model.fit(x1,y)
print(model.intercept_, model.coef_)
#Result: scikit-learn
yp2 = model.predict(x1)
print(yp2)
#Linear regression representation using scatter plot this is the visualization part
plt.scatter(x1,y)
plt.plot(x1,yp2,color='blue')
plt.show()
#Prediction for new values
x1new = pd.DataFrame(np.hstack(np.array([[1],[0],[-0.12],[0.52]])))
x1new.columns=["x1"]
yp2new = model.predict(x1new)
print(yp2new)



#Multiple linear regression
# Input Dataframe
y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])
x1 = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])
x2 = np.array([[1],[0],[1],[1],[0],[1],[0],[1]])

id = np.ones((8,1))
x = np.hstack((id,x1,x2))
print(x)
beta=(np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y))
print(beta)
#Result - Calculation
yp1 = beta[0]+beta[1]*x1+beta[2]*x2
print(np.hstack((x,y,yp1)))
#using multiple linear regression using scikit learn
#Input Dataframe - defining the data
d = pd.DataFrame(np.hstack((x1,x2,y)))
d.columns = ["x1","x2","y"]
print(d)
#step 2 fit 
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
