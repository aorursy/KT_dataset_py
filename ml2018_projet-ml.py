import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd

import math
df = pd.read_csv("../input/train.csv")

train = df.as_matrix()



train_y = train[:,0].astype('int8')

train_x = train[:,1:].astype('float64')



train = None



print("Shape Train Images: (%d,%d)" % train_x.shape)

print("Shape Labels: (%d)" % train_y.shape)
print(train_x.T)
def one_hot(X,C) :

    n = X.shape[0]

    X_hot = np.zeros((n,C))

    for i in range(n) : 

        X_hot[i][X[i]] = 1

    return X_hot

    
Y = one_hot(train_y,10)

def initializing_weights(layer_dims):

    """layer_dims : list containing the number of neurones in each layer 

       Using He initialization 

    """

    parameters={}

    for i in range(1,len(layer_dims)):

        

        

        parameters["W"+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2/layer_dims[i-1])

        parameters["b"+str(i)]=np.zeros((layer_dims[i],1))

    

    return parameters
def linear_Z(X,W,b):

    """ doing linear forward : Z=W*X+b

        Saving parameters for the backprop

    """

    Z = np.dot(W,X)+b

    store= (X,W,b)

    return Z , store



def relu(x) :

    return (x>0)*x

def softmax(x):

    x -= np.max(x)

    sm = (np.exp(x)/ np.sum(np.exp(x),axis=0,keepdims=True))

    

    return sm




def linear_activation(X,W,b, activation):

    """ applying activation and saving Z for the backprop

    """

    if activation == "relu" :

        Z, store = linear_Z(X,W,b)

        A = relu(Z)

    if activation == "softmax":

        Z, store = linear_Z(X,W,b)

        A = softmax(Z)

    cache = (store,Z)

    

    return A, cache
def forward_prop(X,parameters) : 

    

    caches = []

    L = len(parameters) // 2

    A = X

    for l in range(1,L):

        A_0 = A

        W = parameters["W"+str(l)]

        b = parameters["b"+str(l)]

        A , cache = linear_activation(A_0, W, b, activation = "relu")

        caches.append(cache)

    Wl = parameters["W"+str(L)]

    bl = parameters["b"+str(L)]

    A, cache = linear_activation(A,Wl, bl, activation = "softmax")

    caches.append(cache)

    return A, caches
def compute_cost(A,Y) : 

    """ 

    Computing cross entropy cost function

    """

    m = np.shape(Y)[1]

    

    cost = -(1/m) * np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A)))

    cost = np.squeeze(cost)

    return cost
def linear_backprop (dZ, cache):

    

    A, W, b = cache

    m = np.shape(A)[1]

    

    dW = (1/m) * np.dot(dZ,A.T)

    db = (1/m) * np.sum(dZ,axis = 1, keepdims = True )

    dA = np.dot(W.T,dZ)

    

    return dA, dW, db 

    

    
def backward_relu(x) : 

    return (x>0)*1

def backward_softmax(x) : 

    return softmax(x)*(1-softmax(x))
def backward_activation(dA, cache, activation) :

    

    linear_cache, Z = cache

    if activation == 'relu' :

        dZ = dA * backward_relu(Z)

        dA_prev, dW, db =  linear_backprop(dZ,linear_cache)

    if activation == 'softmax' : 

        dZ = dA * backward_softmax(Z)

        dA_prev, dW ,db = linear_backprop(dZ, linear_cache)

    return dA_prev, dW, db
def backpropagation(AL, Y, caches) :

    m =np.shape(AL)[1]

    grads = {}

    L = len(caches) # number of layers

    Y = Y.reshape(AL.shape)

    

    

    dAL = -np.divide(Y,AL) + np.divide(1-Y,1-AL)

    

    current_cache = caches[L - 1]

    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation(dAL, current_cache, activation = "softmax")



    for l in reversed(range(L-1)) :

        

        current_cache = caches[l]

        

        grads["dA"+ str(l+1)], grads["dW" + str(l+1)], grads["db" + str(l+1)]  = backward_activation(grads["dA"+ str(l+2)], current_cache, activation = "relu")

        

    return grads    

    
def initialize_adam(parameters) :

    L = len(parameters) // 2

    v = {}

    s = {}

    for l in range(L) : 

        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])

        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])

        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

    return v, s



        

        
def update_AdamOptimizer(parameters, grads,v,s, learning_rate, t, beta1 = 0.9, beta2 =0.999 , epsilon = 10**-8) : 

    L = len(parameters) // 2 

    s_corrected = {}

    v_corrected = {}

    for l in range(L) :

        v['dW' + str(l+1)] = beta1 * v['dW' + str(l+1)]  + (1-beta1) * grads['dW'+str(l+1)]

        s['dW' + str(l+1)] = beta2 * s['dW' + str(l+1)] + (1 - beta2) * np.square(grads['dW'+str(l+1)])

        v['db' + str(l+1)] = beta1 * v['db' + str(l+1)] + (1-beta1) * grads['db'+str(l+1)]

        s['db' + str(l+1)] = beta2 * s['db' + str(l+1)] + (1 - beta2) * np.square(grads['db'+str(l+1)])

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-np.power(beta1,t))

        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-np.power(beta1,t))

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta2,t))

        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta2,t))

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+epsilon))

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)])+epsilon))

    return parameters, v, s
def random_mini_batches(X, Y, mini_batch_size = 64):

    m = X.shape[1]  

    

    mini_batches = []

        

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation]

    

    num_complete_minibatches = math.floor(m/mini_batch_size) 

    

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]

        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]

        

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

         

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches : m]

        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches : m]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
def model(X, Y, layers_dims, learning_rate = 0.001, mini_batch_size = 32,

          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-7, num_epochs = 50, print_cost = True):

    L = len(layers_dims)             # number of layers in the neural networks

    costs = []                       # to keep track of the cost

    parameters = initializing_weights(layers_dims)  # intializing weights of the network

    v, s = initialize_adam(parameters)  # initializing the momentum for adam optimizer

    t = 0

    cost = 0

    for i in range(num_epochs) : 

        mini_batches = random_mini_batches(X, Y, mini_batch_size = mini_batch_size) 

        for mini_batch in mini_batches :

            mini_X, mini_Y = mini_batch

            AL, caches = forward_prop(mini_X,parameters) 

            cost =compute_cost(AL,mini_Y)

            grads = backpropagation(AL,mini_Y,caches)

            t = t + 1

            parameters, v, s = update_AdamOptimizer(parameters, grads, v = v , s = s ,learning_rate = learning_rate,t = t , epsilon = epsilon)

        

        if print_cost and i % 5 == 0:

            print ("Cost after epoch %i: %f" %(i, cost))

        if print_cost and i % 5 == 0:

            costs.append(cost)

                

    # plot the cost

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('epochs (per 2)')

    plt.title("Learning rate = " + str(learning_rate))

    plt.show()

    return parameters
layers_dims=[784,50,10]

train_x = train_x / 255.

parameters=model(train_x.T,Y.T,layers_dims, mini_batch_size = 32)