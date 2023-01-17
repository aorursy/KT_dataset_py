import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer
def initialiseNetwork(num_features):

  W = np.zeros((num_features, 1))

  b = 0

  parameters = {"W": W, "b": b}

  return parameters
def sigmoid(z):

  a = 1/(1 + np.exp(-z))

  return a
def forwardPropagation(X, Y, parameters):

  W = parameters["W"]

  b = parameters["b"]

  Z = np.dot(W.T,X) + b

  A = sigmoid(Z)

  return A
def cost(A, Y, num_samples):

  cost = -1/num_samples *np.sum(Y*np.log(A) + (1-Y)*(np.log(1-A)))

  return cost
def backPropagration(X, Y, A, num_samples):

  dZ =   A - Y                        

  dW =  (np.dot(X,dZ.T))/num_samples                            #(X dot_product dZ.T)/num_samples

  db =    np.sum(dZ)/num_samples                            #sum(dZ)/num_samples

  return dW, db

  
def updateParameters(parameters, dW, db, learning_rate):

  W =  parameters["W"] - (learning_rate * dW)

  b =  parameters["b"] - (learning_rate * db)

  return {"W": W, "b": b}

  
def model(X, Y, num_iter, learning_rate):

  num_features = X.shape[0]

  num_samples = X.shape[1]

  parameters = initialiseNetwork(num_features)                                                    #call initialiseNetwork()

  for i in range(num_iter):

    A = forwardPropagation(X, Y, parameters)                                                         # calculate final output A from forwardPropagation()

    if(i%100 == 0):

      print("cost after {} iteration: {}".format(i, cost(A, Y, num_samples)))

    dW, db = backPropagration(X, Y, A, num_samples)                                                 # calculate  derivatives from backpropagation

    parameters = updateParameters(parameters, dW, db, learning_rate)                                         # update parameters

  return parameters

    

    

  
def predict(W, b, X):

  Z = np.dot(W.T,X) + b

  Y = np.array([1 if y > 0.5 else 0 for y in sigmoid(Z[0])]).reshape(1,len(Z[0]))

  return Y
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,

                                                   random_state = 25)
def normalize(data):

  col_max = np.max(data, axis = 0)

  col_min = np.min(data, axis = 0)

  return np.divide(data - col_min, col_max - col_min)
X_train_n = normalize(X_train)

X_test_n = normalize(X_test)
X_trainT = X_train_n.T

X_testT = X_test_n.T

y_trainT = y_train.reshape(1,X_train_n.T.shape[1])

y_testT =  y_test.reshape(1,X_testT.shape[1])
parameters = model(X_trainT, y_trainT, 4000, 0.75)                 #call the model() function with parametrs mentioned in the above cell
parameters
yPredTrain = predict(parameters['W'], parameters['b'], X_trainT )   # pass weigths and bias from parameters dictionary and X_trainT as input to the function

yPredTest = predict(parameters['W'], parameters['b'], X_testT)    # pass the same parameters but X_testT as input data
accuracy_train = 100 - np.mean(np.abs(yPredTrain - y_trainT)) * 100

accuracy_test = 100 - np.mean(np.abs(yPredTest - y_testT)) * 100

print("train accuracy: {} %".format(accuracy_train))

print("test accuracy: {} %".format(accuracy_test))

with open("Output.txt", "w") as text_file:

  text_file.write("train= %f\n" % accuracy_train)

  text_file.write("test= %f" % accuracy_test)