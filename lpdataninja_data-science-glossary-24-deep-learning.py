import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Each image is of size 64x64 pixels,
# There are 10 unique signs in the images 0 to 9


x_1 = np.load('../input/Sign-language-digits-dataset/X.npy')
Y_1 = np.load('../input/Sign-language-digits-dataset/Y.npy')

img_size = 64

plt.subplot(1, 2, 1)
plt.imshow(x_1[260].reshape(img_size, img_size))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(x_1[900].reshape(img_size, img_size))
plt.axis('off')
# Sign zero is from 204 to 409 and size one is from 822 to 1027, we'll only select 205 samples from each sign

X = np.concatenate((x_1[204:409], x_1[822:1027] ), axis=0)

z = np.zeros(205)
o = np.ones(205)

Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)

print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
# Now we have three dimensional input arrays X_train and X_test, but we need to 
# flatten them to 2D so that we can feed it to outr Deep Learning model,

X_train = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])  # New shape (348, 4096)
X_test = X_test.reshape(number_of_test, X_test.shape[1]*X_test.shape[2])     # New shape (62, 4096)

print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
# Transposing the values

x_train = X_train.T 
x_test = X_test.T
y_train = Y_train.T
y_test = Y_test.T

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
# Let's initialize our weights and bias

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)  # Fills the array of shape (dimension, 1) with 0.01
    b = 0
    return w, b

w, b = initialize_weights_and_bias(4096)
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

sigmoid(0)
def forward_propogation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]                        # x_train.shape[1]  is for scaling
    return cost
def forward_backward_propagation(w, b, x_train, y_train):
    
    # Forward Propogation (Weights and bias to cost)
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # Backward Propogation (from cost to updating weights and bias)
    derivate_weight = (np.dot(x_train, ((y_head - y_train).T)))/x_train.shape[1]
    derivate_bias = np.sum(y_head - y_train)/x_train.shape[1]
    gradients = {'derivative_weight':derivate_weight, 'derivative_bias':derivate_bias}
    return cost, gradients
# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
 # prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test)
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=42, max_iter=42)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
# Sequential Neural Network
model = Sequential()

# Input Layer
model.add(Dense(units=8, activation='relu', input_dim=x_train.shape[1]))

# Hidden Layer
model.add(Dense(units=4, activation='relu'))

# Output Layer
model.add(Dense(units=1, activation='tanh'))
# Compiling the model (Configuring the model for training)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training
model.fit(X_train, y_train, epochs=100, validation_split=0.2)
# Predictions
preds = model.predict(X_test)