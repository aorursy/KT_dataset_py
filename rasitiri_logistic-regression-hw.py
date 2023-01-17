import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
data.head()
data.info()
data.drop(['PassengerId','Cabin','Name','Sex','Ticket','Embarked','Age'],axis=1,inplace=True)
data.head()
y=data.Survived.values
x_data = data.drop(['Survived'],axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

def iwab(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head
def fbp(w,b,x_train,y_train):
    #Forward
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    #Backward
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                
    gradients = {"Derivative Weight": derivative_weight, "Derivative Bias": derivative_bias}
    
    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        cost,gradients = fbp(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["Derivative Weight"]
        b = b - learning_rate * gradients["Derivative Bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_pre = np.zeros((1,x_test.shape[1]))
    
    #   if z value is bigger than 0.5, our prediction is sign one (y_head=1),
    #   if z value is smaller than 0.5, our prediction is sign zero (y_head=0),
    
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            y_pre[0,i] = 0
        else:
            y_pre[0,i] = 1

    return y_pre
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
  
    dimension =  x_train.shape[0] 
    w,b = iwab(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    print("Test Accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 50)  
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 2, num_iterations = 200)  
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 500)  
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print('Test Accuracy:',lr.score(x_test.T,y_test.T))
