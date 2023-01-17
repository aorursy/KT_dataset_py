import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("../input/voice.csv")
#data.info()
#data.head(10)
data.label = [1 if item == 'male'else 0 for item in data.label]
x = data.drop(['label'], axis=1)
y = data.label.values
x = (x - np.min(x)) / (np.max(x) - np.min(x)).values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
#definition weights and bias
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b
# create sigmoid function for y_head score
def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head
# definition train for our model
def forward_backward_propagation(w,b,x_train, y_train):
    #forward
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = - y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]
    
    #backward
    derivative_weight = (np.dot(x_train,((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weights": derivative_weight, "derivative_bias": derivative_bias}
    return cost,gradients
#update weights and bias
def update(w,b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    for i in range(number_of_iteration):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["derivative_weights"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("iterasyondan sonraki maliyet {} {}".format(i, cost))
    parameters= {"weights": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation = 'vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
#predict test data
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    return Y_prediction
#logistic regression model implement
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    dimension = x_train.shape[0]
    w, b =initialize_weights_and_bias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters['weights'], parameters['bias'], x_test)
    y_prediction_train = predict(parameters['weights'], parameters['bias'], x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iterations=2000)
lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)
lr.score(x_test.T, y_test.T)
