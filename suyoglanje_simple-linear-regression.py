#importing certain libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../input/coursera-ml/ex1data1.txt',header = None) #Read from dataset

X = data.iloc[:,0]; #Read first column
y = data.iloc[:,1]; #Read second column

m = len(y) #number of training examples

data.head()  # view first five rows of data

plt.scatter(X,y)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

X = X[:,np.newaxis] #convert rank 1 array to rank 2 array
y = y[:,np.newaxis]

theta = np.zeros([2,1]) #initialize the initial parameter theta to 0

iterations = 1500

alpha = 0.01 #initialize learning rate to 0.01

ones = np.ones((m,1)) #this ii use as X0 term because X0 = 1

X = np.hstack((ones,X)) #adding intercept term
def ComputeCost(X,y,theta):
    temp = np.dot(X,theta) - y
    return np.sum(np.power(temp , 2)) / (2*m)

J = ComputeCost(X,y,theta)
print(J)
def GradientDescent(X,y,theta, alpha , iterations):
    for _ in range(iterations):
        temp = np.dot(X,theta) - y
        temp = np.dot(X.T , temp)
        theta = theta - (alpha/m) *temp
    return theta

theta = GradientDescent(X,y,theta,alpha,iterations)

print(theta)
J = ComputeCost(X,y,theta)

print(J)
plt.scatter(X[:,1],y)
plt.xlabel('Polpulation of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1],np.dot(X,theta)) #ploting a line
plt.show()
def Predict(X_test):
    profit = np.dot(X_test,theta)
    return profit
predict_data = pd.read_csv('../input/food-truck-profit/predict_profit.txt') #loading prediction data

predict_data = predict_data.iloc[:,0] 

predict_data = predict_data[:,np.newaxis]  #convert rank 1 array to rank 2 array

m_predict = len(predict_data)
ones = np.ones((m_predict , 1))

predict_data = np.hstack((ones,predict_data)) #adding intercept term
predicted_profit = Predict(predict_data)

print(predicted_profit)