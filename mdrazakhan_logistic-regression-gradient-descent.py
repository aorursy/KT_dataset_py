import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/data.csv')
df.head()
df = df.dropna()    #dropping NaN values from dataframe
X = df.iloc[:, 1:3].values #creating a matrix of features
Y = df.iloc[:, 3].values #creating a vector matrix of results
m = 100 #no of cases
def plotx():
    for i in range(m):
        if(Y[i] == 0):
            plt.scatter(X[i][0], X[i][1], c = 'r')
        else:
            plt.scatter(X[i][0], X[i][1], marker = 'x', c = 'b')

    
X_bias = np.ones((m, 1))# creating a matrix of all ones for X_0
X_updated = np.append(X_bias, X, axis = 1) # appending the matrix
Y = Y.reshape([100, 1])#reshaping Y from 100, to 100,1

def h(theta, x):#hypothesis
    
    return g(np.transpose(theta).dot(x))
    
def g(z):#sigmound function
    
    return 1/(1+np.exp(-z))
def j(theta):#cost function
    bloop = 0
    for i in range(m):
        bloop += (Y[i].dot(np.log(h(theta, X_updated[i]))) + (1 - Y[i]).dot(np.log(1 - h(theta, X_updated[i]))))
        
    return -bloop/m
def djd(theta, j):
    bloop = 0
    for i in range(m):
        bloop = bloop + (h(theta, X_updated[i]) - Y[i])*X_updated[i][j]
        
    return bloop/m
def gradient_descent(theta_start, alpha):
    theta_grad = [theta_start]
    j_grad = [j(theta_start)]
        
    while True:
        
        temp = theta_grad[-1]
        for i in range(theta_start.size):
            theta_start[i] = theta_start[i] - alpha*djd(temp, i)
        
        
        theta_grad.append(theta_start)
        j_grad.append(j(theta_start))
        
        if(j_grad[-1]-j_grad[-2] < 10**(-5)):
            break
            
        if(j_grad[-1] < j_grad[-2]):
            alpha /= 3
    
    print('Theta is :',theta_start)
    plotx()
    
    x_1 = np.linspace(X.min(), X.max(), 1000)
    plt.plot(x_1, -(theta_start[1]*x_1 + theta_start[0])/theta_start[2])
    plt.show()
    
    x_predict = np.ones((X_updated[0].size, 1))       
    for i in range(X_updated[0].size-1):
        x_predict[i+1] = float(input('Input X for prediction: ')) 
    y_predict = h(theta_start, x_predict)
    
    print('Probability to be 1 is ',y_predict)
    
gradient_descent(np.zeros((3, 1)), 10)

