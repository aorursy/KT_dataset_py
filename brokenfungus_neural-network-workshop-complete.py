#Import libraries
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

#Load data
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

#Prepare Data
scaler = StandardScaler()
x_train = scaler.fit_transform(xtrain.reshape((-1, 28 * 28)))
x_test = scaler.transform(xtest.reshape((-1, 28 * 28)))

#Prepare Labels
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(ytrain.reshape((-1, 1)))
y_test = encoder.transform(ytest.reshape((-1, 1)))

#Showcase the data
plt.imshow(xtrain[1], cmap='Greys')
#Initialize Weights
w1 = np.random.standard_normal((784, 16))
w2 = np.random.standard_normal((16, 16))
w3 = np.random.standard_normal((16, 10))

#Initialize Biases
b1 = np.random.standard_normal(16)
b2 = np.random.standard_normal(16)
b3 = np.random.standard_normal(10)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def mean_squared_error(y, y_hat):
    return np.mean((y_hat - y)**2).reshape((-1, 1))
def mean_squared_error_derivative(y, y_hat):
    return 2 * np.mean(y_hat - y).reshape((-1, 1))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
lr = - 0.001
batch_size = 300
epochs = 20
minibatch_steps = len(x_train) // batch_size
for epoch in range(epochs):
    loss = 0
    for i in range(minibatch_steps):
        #Get Minibatch
        x = x_train[i * batch_size:i * batch_size + batch_size]
        y = y_train[i * batch_size:i * batch_size + batch_size]

        #Feed Forward
        z1 = x @ w1 + b1
        z2 = sigmoid(z1) @ w2 + b2
        z3 = sigmoid(z2) @ w3 + b3
        y_hat = sigmoid(z3)
        
        #Calculate Loss
        loss += mean_squared_error(y, y_hat).mean()

        #Backpropagation
        b3_derivative = mean_squared_error_derivative(y, y_hat) * sigmoid_derivative(z3)
        w3_derivative = z2.T @ b3_derivative
        
        b2_derivative = (b3_derivative @ w3.T) * sigmoid_derivative(z2)
        w2_derivative = z1.T @ b2_derivative
        
        b1_derivative = (b2_derivative @ w2.T) * sigmoid_derivative(z1)
        w1_derivative = x.T @ b1_derivative

        #Update Weights
        w3 += lr * w3_derivative
        w2 += lr * w2_derivative
        w1 += lr * w1_derivative
        
        b3 += lr * b3_derivative.mean(axis=0)
        b2 += lr * b2_derivative.mean(axis=0)
        b1 += lr * b1_derivative.mean(axis=0)
    
    z1 = (x_train @ w1) + b1
    z2 = sigmoid(z1) @ w2 + b2
    z3 = sigmoid(z2) @ w3 + b3
    y_hat = sigmoid(z3)
    
    loss /= minibatch_steps
    #accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_hat, axis=1))
    print(f"Epoch: {epoch} Loss: {loss:.5f}")