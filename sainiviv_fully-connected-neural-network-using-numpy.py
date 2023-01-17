# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_file = "/kaggle/input/digit-recognizer/train.csv"

test_file = "/kaggle/input/digit-recognizer/test.csv"



raw = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

print (raw.shape)



raw_test = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

print (raw_test.shape)



Y = raw[:,0]     # first column of raw data are image labels

X = raw[:,1:785]   # rest of the pixel data is in columns 1 to 785

Y = Y.reshape(raw.shape[0],1)



print ("Y's shape: " + str(Y.shape))

print ("X's shape: " + str(X.shape))



test_x = raw_test
m_train = X.shape[0]

m_test = test_x.shape[0]



# Transpose train and test data to have training examples column wise and features row wise

# Standardize data to have feature values between 0 and 1.

train_x = (X.T)/255.

test_x = (test_x.T)/255.

train_y = Y.T

print ("train_x's shape: " + str(train_x.shape))

print ("train_y's shape: " + str(train_y.shape))

print ("test_x's shape: " + str(test_x.shape))
#Multi_Y dimensions = [10,42000]

L = 10

multi_y = np.zeros((10,train_x.shape[1]))



for l in range(0, L):

    temp = train_y

    temp = np.where(temp == l, 1, 0)

    multi_y[l,:] = temp

        

print(multi_y.shape)
i = 0

L = 2000

batchSize = 512

costsFor = []

costsPrint = []



while i < (train_x.shape[1]/batchSize)-1:

    XB = train_x[:,i*batchSize : (i+1)*batchSize]

    YB = multi_y[:,i*batchSize : (i+1)*batchSize]

    #print ("Shape of XB "+str(i)+"= :" + str(YB.shape),end="  ")

    #print ("Shape of YB "+str(i)+"= :" + str(YB.shape))

    for l in range(0, L):

        cost = l

        costsFor.append(cost)

    

        if (i*L+l) % 100 == 0:

            costsPrint.append(cost)

        

    i = i +1
def rand_disp (X,Y,row = 3, col = 3):



    fig, axes = plt.subplots(row,col, figsize=(12,9))

    axes = axes.flatten()

    idx = np.random.randint(0,X.shape[1]-1,size=row*col)   

    for i in range(row*col):

        axes[i].imshow(X[:,idx[i]].reshape(28,28), cmap='gray')

        axes[i].axis('off') # hide the axes ticks

        axes[i].set_title(str(int(Y[:,idx[i]])), color= 'black', fontsize=25)

    plt.show()

    

    return
rand_disp(train_x,train_y,3,3)
def initialize_parameters_deep(layer_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the dimensions of each layer in our network

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])

                    bl -- bias vector of shape (layer_dims[l], 1)

    """

    

    np.random.seed(3)

    parameters = {}

    L = len(layer_dims)            # number of layers in the network



    for l in range(1, L):

       

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

       

        

        print("W" + str(l) + "= " + str(parameters['W'+ str(l)].shape))

        print("b" + str(l) + "= " + str(parameters['b' + str(l)].shape))

        

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))

        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))



        

    return parameters
def sigmoid(Z):

    """

    Implements the sigmoid activation in numpy

    

    Arguments:

    Z -- numpy array of any shape

    

    Returns:

    A -- output of sigmoid(z), same shape as Z

    cache -- returns Z as well, useful during backpropagation

    """

    

    A = 1/(1+np.exp(-Z))

    cache = Z

    

    return A, cache
def relu(Z):

    """

    Implement the RELU function.



    Arguments:

    Z -- Output of the linear layer, of any shape



    Returns:

    A -- Post-activation parameter, of the same shape as Z

    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently

    """

    

    A = np.maximum(0,Z)

    

    assert(A.shape == Z.shape)

    

    cache = Z 

    return A, cache
def sigmoid_backward(dA, cache):

    """

    Implement the backward propagation for a single SIGMOID unit.



    Arguments:

    dA -- post-activation gradient, of any shape

    cache -- 'Z' where we store for computing backward propagation efficiently



    Returns:

    dZ -- Gradient of the cost with respect to Z

    """

    

    Z = cache

    

    s = 1/(1+np.exp(-Z))

    dZ = dA * s * (1-s)

    

    assert (dZ.shape == Z.shape)

    

    return dZ
def relu_backward(dA, cache):

    """

    Implement the backward propagation for a single RELU unit.



    Arguments:

    dA -- post-activation gradient, of any shape

    cache -- 'Z' where we store for computing backward propagation efficiently



    Returns:

    dZ -- Gradient of the cost with respect to Z

    """

    

    Z = cache

    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    

    # When z <= 0, you should set dz to 0 as well. 

    dZ[Z <= 0] = 0

    

    assert (dZ.shape == Z.shape)

    

    return dZ

def linear_forward(A, W, b):

    """

    Implement the linear part of a layer's forward propagation.



    Arguments:

    A -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)



    Returns:

    Z -- the input of the activation function, also called pre-activation parameter 

    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """

  

    Z = np.dot(W,A) + b



    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    

    return Z, cache
def linear_activation_forward(A_prev, W, b, activation):

    """

    Implement the forward propagation for the LINEAR->ACTIVATION layer



    Arguments:

    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)

    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"



    Returns:

    A -- the output of the activation function, also called the post-activation value 

    cache -- a python tuple containing "linear_cache" and "activation_cache";

             stored for computing the backward pass efficiently

    """

    

    if activation == "sigmoid":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = sigmoid(Z)

    

    elif activation == "relu":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = relu(Z)

    

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)



    return A, cache
def L_model_forward(X, parameters):

    """

    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    

    Arguments:

    X -- data, numpy array of shape (input size, number of examples)

    parameters -- output of initialize_parameters_deep()

    

    Returns:

    AL -- last post-activation value

    caches -- list of caches containing:

                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)

    """



    caches = []

    A = X

    

    # number of layers in the neural network

    L = len(parameters) // 2    # As there are both weights and biases in each layer hence a devide by 2

    

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.

    for l in range(1, L):

        A_prev = A 

        W = parameters["W" + str(l)]

        b = parameters["b" + str(l)]

        A, cache = linear_activation_forward(A_prev, W, b, activation = "relu")

        caches.append(cache)

    

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    W = parameters["W" + str(L)]

    b = parameters["b" + str(L)]   

    AL, cache = linear_activation_forward(A, W, b, activation = "sigmoid")

    caches.append(cache)

    

    assert(AL.shape == (10,X.shape[1]))

            

    return AL, caches
def compute_cost(AL, Y):

    """

    Implement the cost function defined by equation (7).



    Arguments:

    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)

    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)



    Returns:

    cost -- cross-entropy cost

    """

    

    m = Y.shape[1]



    # Compute loss from aL and y.

    cost = -(1/m)* np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1 - AL)))

    

    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    #assert(cost.shape == ())

    

    return cost
def compute_cost_with_regularization(AL, Y, parameters, lambd =0):

    

    cross_entropy_cost = compute_cost(AL, Y) # This gives you the cross-entropy part of the cost

    m = Y.shape[1]

    L = len(layer_dims)

    tmp = 0

    for l in range(1, L):

        W = parameters["W" + str(l)]

        tmp = tmp+np.sum(np.square(W))

        

    L2_regularization_cost = (lambd)/(2*m) * tmp

    cost = cross_entropy_cost + L2_regularization_cost

    

    return cost
def linear_backward(dZ, cache,lambd =0.):

    """

    Implement the linear portion of backward propagation for a single layer (layer l)



    Arguments:

    dZ -- Gradient of the cost with respect to the linear output (of current layer l)

    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer



    Returns:

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

    dW -- Gradient of the cost with respect to W (current layer l), same shape as W

    db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    A_prev, W, b = cache

    m = A_prev.shape[1]

   

    dW = 1/m*np.dot(dZ,cache[0].T) + lambd/m * W

    db = 1/m*np.sum(dZ,axis=1,keepdims=True)

    dA_prev = np.dot(cache[1].T,dZ)

    

    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    

    return dA_prev, dW, db
def linear_activation_backward(dA, cache, lambd =0, activation = "sigmoid"):

    """

    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    

    Arguments:

    dA -- post-activation gradient for current layer l 

    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently

    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    

    Returns:

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

    dW -- Gradient of the cost with respect to W (current layer l), same shape as W

    db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    linear_cache, activation_cache = cache

    

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)

        

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)

    

    return dA_prev, dW, db
def L_model_backward(AL, Y, caches,lambd):

    """

    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    

    Arguments:

    AL -- probability vector, output of the forward propagation (L_model_forward()) 

    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)

    caches -- list of caches containing:

                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)

                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    

    Returns:

    grads -- A dictionary with the gradients

             grads["dA" + str(l)] = ... 

             grads["dW" + str(l)] = ...

             grads["db" + str(l)] = ... 

    """

    grads = {}

    L = len(caches) # the number of layers

    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    

    # Initializing the backpropagation

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

    current_cache = caches[L - 1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =  linear_activation_backward(dAL, current_cache, lambd, activation = "sigmoid")

    

    # Loop from l=L-2 to l=0

    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients.

        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, lambd, activation = "relu")

        grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

      

    return grads
def update_parameters(parameters, grads, learning_rate):

    """

    Update parameters using gradient descent

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    grads -- python dictionary containing your gradients, output of L_model_backward

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

                  parameters["W" + str(l)] = ... 

                  parameters["b" + str(l)] = ...

    """

    

    L = len(parameters) // 2 # number of layers in the neural network



    # Update rule for each parameter. Use a for loop.

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters
def L_layer_model(X, Y, layers_dims, learning_rate = 0.4, num_iterations = 2000, print_cost=False,lambd = 0,batch_size=1):

    """ 

    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    

    Arguments:

    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).

    learning_rate -- learning rate of the gradient descent update rule

    num_iterations -- number of iterations of the optimization loop

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """



    np.random.seed(1)

    costs = []              # keep track of cost

    

    # Parameters initialization.

    parameters = initialize_parameters_deep(layers_dims)

    

    m = X.shape[1]       # number of training examples

        

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation]

    

   

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        k = 0

        cost_total = 0

        cost_avg = 0.0

        XBatches = shuffled_X[:,k*batch_size : (k+1)*batch_size]

        YBatches = shuffled_Y[:,k*batch_size : (k+1)*batch_size]

        

        while k < (train_x.shape[1]/batch_size):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

            AL, caches = L_model_forward(XBatches, parameters)



            # Compute cost.

            #cost_total += compute_cost(AL, YBatches)

            cost_total = compute_cost_with_regularization(AL, YBatches, parameters, lambd)

        

            # Backward propagation.

            grads = L_model_backward(AL, YBatches, caches,lambd)

            

            # Update parameters.

            parameters = update_parameters(parameters, grads, learning_rate)



            k = k+1

            

            cost_avg = (cost_avg * (k-1)/k) + cost_total/k

        

        # Print the cost every 100 epoch

        interval = 50

        if print_cost and i % interval == 0:

            print ("Cost after epoch %i: %f" %(i, cost_avg))

        if print_cost and i % interval == 0:

            costs.append(cost_avg)

        

        

    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per hundreds)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()



    return parameters
def predict(X, parameters):

    """

    This function is used to predict the results of a  L-layer neural network.

    

    Arguments:

    X -- data set of examples you would like to label

    parameters -- parameters of the trained model

    

    Returns:

    p -- predictions for the given dataset X

    """

    #Shape of X = (784, 1)

    

    m = X.shape[1]  # m = 1 

    n = len(parameters) // 2 # number of layers in the neural network

    p = np.zeros((10,m))



    # Forward propagation

    probas, caches = L_model_forward(X, parameters)

    

    # convert probas to 0/1 predictions

    for j in range(0, probas.shape[1]): # j = 0 -> 42000

        for i in range(0, probas.shape[0]): # i = 0 -> 10

            #if probas[i,j] >= 0.5:

            if probas[i,j] >= np.max(probas[:,j]): 

                p[i,j] = 1

            else:

                p[i,j] = 0

    

    return p
def accuracy (Y_truth, Predict):

  

    m = Y_truth.shape[1] 

    s = 0

    for i in range(0, m):  

        comparison = Y_truth[:,i] == Predict[:,i]

        equal_arrays = comparison.all()

        if equal_arrays == True:    

            s = s+1

    acc = s/m

    return acc
def getLabels (predict_Vec):



    label = np.zeros((1,predict_Vec.shape[1]))

    for i in range(0, predict_Vec.shape[1]):

        tmp =  0

        tmp = (np.where(predict_Vec[:,i] == 1)[0])

        if tmp.size == 0:

            label[0,i] = 0

        else:

            label[0,i] = np.squeeze(tmp[0])



    return label
layer_dims = [784, 256,128,10] #  3-layer model
import time

start_time = time.process_time()



parameters = L_layer_model(train_x, multi_y, layer_dims, num_iterations = 2000, print_cost = True,lambd = 0.0,batch_size=train_x.shape[1])

# With batch gradient descents I can observe exploding gradients in later iterations

#parameters = L_layer_model(train_x, multi_y, layer_dims, num_iterations = 1000, print_cost = True,lambd = 0.1,batch_size=128)



end_time = time.process_time()

print ("\n ----- Computation time = " + str((end_time - start_time)) + "sec")
pred_train = predict(train_x, parameters)

acc = accuracy(multi_y, pred_train)

print("Accuracy: "  + str(acc))



train_label = getLabels (pred_train)

rand_disp(train_x,train_label)

pred_test = predict(test_x,parameters)

print(pred_test.shape)

test_label = getLabels (pred_test)
rand_disp(test_x,test_label,4,5)
mydir = "/kaggle/working/"



filelist = [ f for f in os.listdir(mydir) if f.endswith(".csv") ]

for f in filelist:

    os.remove(os.path.join(mydir, f))

    print("removed old submission file -"+ str(os.path.join(mydir, f)))



# Creating pandas dataframe from numpy array

submission = pd.DataFrame({'ImageId': range(1,(np.squeeze(test_label.shape[1])+1)),'Label': test_label[0, :]})

submission.to_csv("mySubmission.csv",index=False)

print(submission)



# mySubmission file looks similar in structure to sample submission but it gives a score of 0.000

# When manually copied the labels from mySubmission file to sample_submission.csv file 

# depending on the choice of hyperperameters a score of ~0.97 is achieved 



# Not sure what is wrong with mySubmission file creation 