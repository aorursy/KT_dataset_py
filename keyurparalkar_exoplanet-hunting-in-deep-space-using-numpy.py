import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#training set
data_train = pd.read_csv('../input/exoTrain.csv')
#test set
data_test = pd.read_csv('../input/exoTest.csv')
# 1 = Exoplanet 0= not exoplanet

Y_train = np.reshape(np.array(data_train['LABEL']),(1,data_train.shape[0]))
Y_train[[Y_train==1]]=0
Y_train[[Y_train==2]]=1
X_train = np.transpose(np.array(data_train[data_train.columns[1:]]))
print(Y_train)
Y_test = np.reshape(np.array(data_test['LABEL']),(1,data_test.shape[0]))
X_test = np.transpose(np.array(data_test[data_test.columns[1:]]))
#shapes of all X and Y for train and test set:
print("Shapes:")
print("Y_train = ",Y_train.shape)
print("X_train = ",X_train.shape)
print("Y_test = ",Y_test.shape)
print("X_test = ",X_test.shape)
# Normalization of both train and test set X:
#training set:
mean_train = np.reshape(np.mean(X_train,axis=0),(1,X_train.shape[1]))
std_train = np.reshape(np.std(X_train,axis=0),(1,X_train.shape[1]))
X_train = (X_train - mean_train)/std_train

#test set:
mean_test = np.reshape(np.mean(X_test,axis=0),(1,X_test.shape[1]))
std_test = np.reshape(np.std(X_test,axis=0),(1,X_test.shape[1]))
X_test = (X_test - mean_test)/std_test
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def defining_structure(X):
    n_i = X.shape[0]
    n_h1 = 12
    n_h2 = 12
    n_o = 1
    
    nodes ={
        "n_i":n_i,
        "n_h1":n_h1,
        "n_h2":n_h2,
        "n_o":n_o
    }
    
    return nodes
nodes = defining_structure(X_train)
print(nodes["n_h1"],nodes["n_h2"])
def random_initialization(X):
    np.random.seed(2)
    
    nodes = defining_structure(X)
    
    W1 = np.random.randn(nodes["n_h1"],nodes["n_i"])*0.01
    b1 = np.zeros((nodes["n_h1"],1))
    W2 = np.random.randn(nodes["n_h2"],nodes["n_h1"])*0.01
    b2 = np.zeros((nodes["n_h2"],1))
    W3 = np.random.randn(nodes["n_o"],nodes["n_h2"])*0.01
    b3 = np.zeros((nodes["n_o"],1))
    
#     print(W1.shape)
#     print(b1.shape)
#     print(W2.shape)
#     print(b2.shape)
#     print(W3.shape)
#     print(b3.shape)

    params = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3
    }
    
    return params
random_initialization(X_train)
#forward propagation:
def forward_prop(X,Y,parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2,
        "Z3":Z3,
        "A3":A3
    }
    
    return cache
    
forward_prop(X_train,Y_train,random_initialization(X_train))
def compute_cost(A3,Y):
    m = Y.shape[1]
    cost = -(np.dot(Y,np.transpose(np.log(A3))) + np.dot((1-Y),np.transpose(np.log(1-A3))))/m
    return cost
def backward_prop(X,Y,parameters,cache):
    m = Y.shape[1]
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    Z3 = cache["Z3"]
    A3 = cache["A3"]
    
    dA3 = (-Y/A3) + (1-Y)/(1-A3)
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,np.transpose(A2))/m
    db3 = np.sum(dZ3,axis=1,keepdims=True)/m
    
    dA2 = np.dot(np.transpose(W3),dZ3)      #dA2 = (12,5087)
    dZ2 = dA2*(1-np.power(np.tanh(Z2),2))
    dW2 = np.dot(dZ2,np.transpose(A1))/m
    db2 = np.sum(dZ2,keepdims=True)/m
    
    dA1 = np.dot(W2,Z2)
    dZ1 = dA1*(1-np.power(np.tanh(Z1),2))
#     dw1 = dz1.X
#     (12,3197) = (12,5087).(3197,5087)
#     dw1 = dz1.np.transpose(X)
#     (12,3197) = (12,5087).(5087,3197)
    dW1 = np.dot(dZ1,np.transpose(X))/m
    db1 = np.sum(dZ1,keepdims=True)/m
    
    grads = {
        "dW3":dW3,
        "dW2":dW2,
        "dW1":dW1,
        "db3":db3,
        "db2":db2,
        "db1":db1
    }
    
    return grads
def update_weh(learning_rate,parameters,grads):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    dW3 = grads["dW3"]
    dW2 = grads["dW2"]
    dW1 = grads["dW1"]
    db3 = grads["db3"]
    db2 = grads["db2"]
    db1 = grads["db1"]
    
    W3 = W3 - learning_rate*dW3
    W2 = W2 - learning_rate*dW2
    W1 = W1 - learning_rate*dW1
    b3 = b3 - learning_rate*db3
    b2 = b2 - learning_rate*db2
    b1 = b1 - learning_rate*db1
    
    params = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3
    }
    
    return params
def model(X,Y,learning_rate=0.03,num_iteration=2000):
    
    parameters = random_initialization(X)
    all_cost = list()
    
    for i in range(num_iteration):
        cache = forward_prop(X,Y,parameters)
        cost = compute_cost(cache["A3"],Y)
        all_cost.append(cost)
        print("Cost for iteration ",i," = ",cost,end='\r')
        
        grads = backward_prop(X,Y,parameters,cache)
        parameters = update_weh(learning_rate,parameters,grads)
        
    nn_model={
        "grads":grads,
        "cache":cache,
        "parameters":parameters,
        "cost":all_cost
    }
    return nn_model
    
nn_model = model(X_train,Y_train)
Y_predict = np.squeeze(nn_model["cache"]["A3"])
for temp in Y_predict:
    print("\n",temp)
    

cost = np.squeeze(nn_model["cost"])
plt.plot(cost)
plt.show()
