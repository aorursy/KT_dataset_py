# Package imports
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets
import sklearn.linear_model


%matplotlib inline

np.random.seed(1) # set a seed so that the results are consistent

import pandas as pd
from sklearn.utils import shuffle
#reading data
df = pd.read_csv('../input/Iris.csv',index_col = 0, skiprows=0 )
#producing training set (XX and y)
d= shuffle(df)
d1 = d[0:125]
d2 = d[126:150]
a =d1.as_matrix(columns=None)
b = np.asmatrix(d1.as_matrix(columns=None))
Y = b[:125, 4]
y = np.zeros((3,125))
for i in range(0,125):
    if(Y[i,0]== 'Iris-setosa' ):
        y[0,i]= 1
    elif(Y[i,0] == 'Iris-versicolor'):
        y[1,i]=1
    else:
        y[2,i]=1
y = np.array((y[:, :]), dtype=np.float)
Xl = b[:125, 0:4]
XX = np.array((Xl[:, :]), dtype=np.float)
XX =XX.T


#producing test set (XXnext and Ynext)

anext =d1.as_matrix(columns=None)
bnext = np.asmatrix(anext)
Ynext = bnext[:25, 4]
ynext = np.zeros((3,25))
for i in range(0,25):
    if(Ynext[i,0]== 'Iris-setosa' ):
        ynext[0,i]= 1
    elif(Ynext[i,0] == 'Iris-versicolor'):
        ynext[1,i]=1
    else:
        ynext[2,i]=1
ynext = np.array((ynext[:, :]), dtype=np.float)
Xnext = bnext[:25, 0:4]
XXnext = np.array((Xnext[:, :]), dtype=np.float)
XXnext = XXnext.T
def layer_sizes(X, Y):
    
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)
def sigmoid(z):
    s = 1/(1+ np.exp(-z))
    return s
def initialize_parameters(n_x, n_h, n_y):
   
    np.random.seed(2) 
 
    W1 = np.random.randn(n_h, n_x) * 0.001
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.001
    b2 = np.zeros((n_y, 1))

    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
def forward_propagation(X, parameters):
  
    # Retrieve each parameter from the dictionary "parameters"
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    
    assert(A2.shape == (3, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
def compute_cost(A2, Y, parameters):

    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs =np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = - np.sum(logprobs)/m
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
        
    # Retrieve also A1 and A2 from dictionary "cache".
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    
    dZ2 = A2 - Y
    dW2 =np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2, axis =1, keepdims = True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1- A1**2)
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1, axis =1, keepdims = True) / m
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
def update_parameters(parameters, grads, learning_rate = 10):
   
    # Retrieve each parameter from the dictionary "parameters"
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    
    # Retrieve each gradient from the dictionary "grads"
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    
    # Update rule for each parameter
    
    W1 = W1 - learning_rate* dW1
    b1 = b1 - learning_rate *db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
       
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache =forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        
        
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

nn_model(XX, y, 4, num_iterations = 10000, print_cost=True)
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    pred = np.ones((3, X.shape[1]))
    pred = pred*3
    for i in range(0,X.shape[1]):
        pred[np.argmax(A2[:,i]), i] = 1
    #for i in range(0,150):     
      #  mn =np.mean((p[i]==y[i])*100)
    mn=0
    for i in range(0,3):
        for j in range(0,X.shape[1]):
            if(pred[i,j]== y[i,j]):
                mn +=1
    
    return mn *100 /X.shape[1]
    
    
parameters = nn_model(XX, y, n_h = 4, num_iterations = 10000, print_cost=True)
t = predict(parameters, XX)
t
parameters = nn_model(XX, y, n_h = 4, num_iterations = 10000, print_cost=True)
t = predict(parameters, XXnext)
t
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(XX, y, n_h, num_iterations = 5000)
    predictions = predict(parameters, XX)
    print(predictions)
