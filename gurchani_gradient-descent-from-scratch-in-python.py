import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import sys
MaxIterations = 2000 #Number of the times thetas will change

alpha = 0.01 

CostArray = [] #This will store the cost of each set of thetas 
data = pd.read_csv('../input/ex1data2.csv', sep=",", header=None)

numberOfColumns = data.shape[0]

thetas = [0]*len(data.columns)
    

def hypothesis(theta, Xaxis):

    thetaArray = np.matrix(np.array(theta)) 

    Xaxis = np.matrix(Xaxis)

    xtrans = np.transpose(Xaxis) 

    mat =  np.matmul(thetaArray, xtrans)

    return mat
def costFunction(thetas, Xaxis, Yaxis):

    resultingMatrix = hypothesis(thetas, Xaxis) - np.matrix(Yaxis)

    totalSum = np.sum(np.square(resultingMatrix))

    totalCost = totalSum / (2*(numberOfColumns))

    CostArray.append(totalCost)

    return totalCost

def updateThetas(theta):

    temp = np.matrix(np.array(theta))

    resultingMatrix = hypothesis(theta, Xaxis) - np.matrix(Yaxis)

    X2 = np.matrix(Xaxis)

    multiplier = np.matmul(resultingMatrix, X2)

    temp = np.matrix(np.array(theta)) - ((alpha/(numberOfColumns))* multiplier)

    global thetas

    thetas = temp 
ones = pd.Series([1]*(data.shape[0]))



Xaxis = (data.iloc[:, :-1] - np.mean(data.iloc[:, :-1]))

Xaxis = Xaxis/np.std(data.iloc[:, :-1])

Xaxis = pd.concat([ones, Xaxis], axis=1)

Yaxis = data[data.columns[-1]]



for j in range(MaxIterations):

    updateThetas(thetas)

    CostArray.append(costFunction(thetas, Xaxis, Yaxis))

print(CostArray[1999])

print(thetas)
plt.plot(CostArray)

plt.show()