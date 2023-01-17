# Imports

import numpy as np

import pandas as pd

import pickle

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



from sklearn.utils import resample

df = pd.read_csv("../input/hr-analytics-case-study/general_data.csv")
print("Dataframe shape is",df.shape)

print("\n",df.head(5))

print("\n ******* \n")

print(df.tail(5))
df.describe()
df.columns
df.isna().sum()
df["NumCompaniesWorked"].fillna(df["NumCompaniesWorked"].mean(),inplace=True)

df["TotalWorkingYears"].fillna(df["TotalWorkingYears"].mean(),inplace=True)
df.isna().sum()
df.drop(['EmployeeCount','EmployeeID','StandardHours','Over18'],axis=1,inplace=True)
df.columns
yes = df[df["Attrition"]=="Yes"]

no = df[df["Attrition"]=="No"]



print("Total number of examples are \n", len(df))

print("Number of yes:",len(yes), "\t Number of no:", len(no), "\t Total is:",len(yes+no))



sns.countplot(x = "Attrition",data=df)

plt.show()
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

df['BusinessTravel'] = labelEncoder_X.fit_transform(df['BusinessTravel'])

df['Department'] = labelEncoder_X.fit_transform(df['Department'])

df['EducationField'] = labelEncoder_X.fit_transform(df['EducationField'])

df['Gender'] = labelEncoder_X.fit_transform(df['Gender'])

df['JobRole'] = labelEncoder_X.fit_transform(df['JobRole'])

df['MaritalStatus'] = labelEncoder_X.fit_transform(df['MaritalStatus'])
# Attriton is dependent var

label_encoder_y = LabelEncoder()

df['Attrition'] = label_encoder_y.fit_transform(df['Attrition'])
df_majority = df[df["Attrition"]==0]

df_minority = df[df["Attrition"]==1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=3699,    # to match majority class

                                 random_state=0) # reproducible results

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])



yes = df_upsampled[df_upsampled["Attrition"]==1]

no = df[df["Attrition"]==0]



print("Total number of examples are \n", len(df_upsampled))

print("Number of yes:",len(yes), "\t Number of no:", len(no), "\t Total is:",len(yes+no))



sns.countplot(x = "Attrition",data=df_upsampled)

plt.show()
print("Dataframe shape is",df_upsampled.shape)

df_upsampled.head()
y = df_upsampled['Attrition']

X = df_upsampled.drop('Attrition', axis = 1)

print(y.shape, X.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)



X = np.array(X)

y = np.array(y)
print(X.shape)

print(y.shape)
def initialize_parameters_deep(layer_dims, n):

    """

    This function takes the numbers of layers to be used to build our model as

    input and otputs a dictonary containing weights and biases as parameters

    to be learned during training

    The number in the layer_dims corresponds to number of neurons in 

    corresponding layer



    @params



    Input to this function is layer dimensions

    layer_dims = List contains number of neurons in one respective layer

                 and [len(layer_dims) - 1] gives L Layer Neural Network

    

    Returns:

    

    parameters = Dictionary containing parameters "W1", "b1", ., "WL", "bL"

                 where Wl = Weight Matrix of shape (layer_dims[l-1],layer_dims[l])

                       bl = Bias Vector of shape (1,layer_dims[l])

    """

    # layers_dims = [19, 7, 5, 1] #  3-layer model

    np.random.seed(3)

    parameters = {}

    L = len(layer_dims)            # Number of layers in the network



    for l in range(1, L):          # It starts with 1 hence till len(layer_dims)

        # Initialize weights randomly according to Xavier initializer in order to avoid linear model

        parameters['W' + str(l)] = np.random.randn(layer_dims[l-1],layer_dims[l])*np.sqrt(n / layer_dims[l-1])

        # Initialize bias vector with zeros

        parameters['b' + str(l)] = np.zeros((1,layer_dims[l]))

        # Making sure the shape is correct

        assert(parameters['W' + str(l)].shape == (layer_dims[l-1], layer_dims[l]))

        assert(parameters['b' + str(l)].shape == (1,layer_dims[l]))



    # parameters = {"W [key]": npnp.random.randn(layer_dims[l-1],layer_dims[l]) [value]}

    return parameters
def sigmoid(Z):

    """

    This function takes the forward matrix Z (Output of the linear layer) as the

    input and applies element-wise Sigmoid activation



    @params



    Z = numpy array of any shape

    

    Returns:



    A = Output of sigmoid(Z), same shape as Z, for the last layer this A is the

        output value from our model



    cache = Z is cached, this is useful during backpropagation

    """

    

    A = 1/(1+np.exp(-Z)) # Using numpy apply sigmoid to Z 

    cache = Z            # Cache the matrix Z

    

    return A, cache



def relu(Z):

    """

    This function takes the forward matrix Z as the input and applies element 

    wise Relu activation



    @params



    Z = Output of the linear layer, of any shape



    Returns:



    A = Post-activation parameter, of the same shape as Z

    cache = Z is cached, this is useful during backpropagation

    """

    

    A = np.maximum(0,Z) # Element-wise maximum of array elements

    # Making sure shape of A is same as shape of Z

    assert(A.shape == Z.shape)

    

    cache = Z           # Cache the matrix Z



    return A, cache





def relu_backward(dA, cache):

    """

    This function implements the backward propagation for a single Relu unit



    @params



    dA = post-activation gradient, of any shape

    cache = Retrieve cached Z for computing backward propagation efficiently



    Returns:



    dZ = Gradient of the cost with respect to Z

    """

    

    Z = cache

    dZ = np.array(dA) # Just converting dz to a correct object.

    

    # When z <= 0, you set dz to 0 as well, as relu sets negative values to 0 

    dZ[Z <= 0] = 0

    # Making sure shape of dZ is same as shape of Z

    assert (dZ.shape == Z.shape)

    

    return dZ



def sigmoid_backward(dA, cache):

    """

    This function implements the backward propagation for a single Sigmoid unit



    @params



    dA = post-activation gradient, of any shape

    cache = Retrieve cached Z for computing backward propagation efficiently



    Returns:

    dZ = Gradient of the cost with respect to Z

    """

    

    Z = cache

    

    s = 1/(1+np.exp(-Z)) # Using numpy apply Sigmoid to Z 

    dZ = dA * s * (1-s)  # This is derivatie of Sigmoid function



    # Making sure shape of dZ is same as shape of Z

    assert (dZ.shape == Z.shape)

    

    return dZ
def linear_forward(A, W, b):

    """

    This function implements the forward propagation equation Z = WX + b



    @params



    A = Activations from previous layer (or input data),

        shape = (number of examples, size of previous layer)

    W = Weight matrix of shape (size of previous layer,size of current layer)

    b = Bias vector of shape (1, size of the current layer)



    Returns:



    Z = The input of the activation function, also called pre-activation parameter

        shape = (number of examples, size of current layer)

    cache = Tuple containing "A", "W" and "b"; 

            stored for computing the backward pass efficiently

    """

    # A = [(3528, 19)], W = [19,7], Z = [3528,7]

    # print(A.shape, W.shape)

    Z = A.dot(W) + b # Here b gets broadcasted 

    #print(Z)

    # Making sure shape of Z = (number of examples, size of current layer)

    assert(Z.shape == (A.shape[0], W.shape[1]))



    cache = (A, W, b) # Cache all the three params 

    

    return Z, cache
def linear_activation_forward(A_prev, W, b, activation):

    """

    This function implements forward propagation LINEAR -> ACTIVATION layer



    @params



    A_prev = Activations from previous layer (or input data), 

             shape = (number of examples, size of previous layer)

    W = Weight matrix of shape (size of previous layer,size of current layer)

    b = Bias vector of shape (1, size of the current layer)

    activation = The activation to be used in this layer, 

                 stored as a text string: "sigmoid" or "relu"



    Returns:



    A = The output of the activation function, also called the post-activation value 

    cache = Tuple containing "linear_cache" and "activation_cache";

            stored for computing the backward pass efficiently

    """

    

    if activation == "sigmoid":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache"

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = sigmoid(Z)

    

    elif activation == "relu":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache"

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = relu(Z)

    # Making sure shape of A = (number of examples, size of current layer)

    assert (A.shape == (A_prev.shape[0],W.shape[1]))

    cache = (linear_cache, activation_cache)

    #print(cache)

    return A, cache
def L_model_forward(X, parameters):

    """

    This function implements forward propagation as following:

    [LINEAR->RELU]*(L-1) -> LINEAR -> SIGMOID computation

    So we apply Relu to all the hidden layers and Sigmoid to the output layer



    @params



    X = Data, numpy array of shape (number of examples, number of features)

    parameters = Output of initialize_parameters_deep() function

    

    Returns:



    AL = last post-activation value, also rferred as prediction from model

    caches = list of caches containing:

             every cache of linear_activation_forward() function

             (there are L-1 of them, indexed from 0 to L-1)

    """



    caches = []

    A = X

    L = len(parameters) // 2            # Number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.

    for l in range(1, L):

        A_prev = A

        # For hidden layers use Relu activation

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')

        #print("A",A.shape)

        caches.append(cache)

    

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    # For output layer use Sigmoid activation

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')

    #print("AL",AL.shape)

    caches.append(cache)

    

    # Making sure shape of AL = (number of examples, 1)

    assert(AL.shape == (X.shape[0],1))

            

    return AL, caches
def compute_cost(AL, Y, parameters, lambd):

    """

    This function implements the Binary Cross-Entropy Cost alng with L2 regularization

    J = -(1/m)*(ylog(predictions)+(1−y)log(1−predictions)) + (λ/2*m)∑(W**2)

    

    @params



    AL = Probability vector corresponding to our label predictions 

         shape =  (number of examples, 1)

    Y  = Ground Truth/ True "label" vector (containing classes 0 and 1) 

         shape  = (number of examples, 1)

    parameters = Dictionary containing parameters as follwoing:

                    parameters["W" + str(l)] = Wl

                    parameters["b" + str(l)] = bl

    lambd = Regularization parameter, int



    Returns:



    cost = Binary Cross-Entropy Cost with L2 Regularizaion 

    """

    

    m = Y.shape[0]  # Number of training examples



    # Compute loss from aL and y

    cross_entropy_cost = -(1/m)*(np.dot(np.log(AL).T,Y) + np.dot(np.log(1-AL).T,(1-Y)))

    #print(cost)

    reg_cost = []

    W = 0

    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1, L+1):

        W = parameters["W" + str(l)]

        reg_cost.append(lambd*1./(2*m)*np.sum(W**2))

        

    cross_entropy_cost = np.squeeze(cross_entropy_cost) # To make sure cost's is scalar (e.g. this turns [[cost]] into cost)

    assert(cross_entropy_cost.shape == ())

    cost = cross_entropy_cost + np.sum(reg_cost)



    return cost
def linear_backward(dZ, cache, lambd):

    """

    This function implements the linear portion of backward propagation for a 

    single layer (layer l)



    @params



    dZ = Gradient of the cost with respect to the linear output of current 

         layer l, shape = (number of examples, size of current layer)

    cache = Tuple of values (A_prev, W, b) coming from the forward propagation 

            in the current layer

    lambd = Regularization parameter, int

    

    Returns:



    dA_prev = Gradient of the cost with respect to the activation of the 

              previous layer l-1, 

              same shape as A_prev(number of examples, size of previous layer)

    dW = Gradient of the cost with respect to W of current layer l, 

         same shape as W(size of previous layer,size of current layer)

    db = Gradient of the cost with respect to b of current layer l, 

         same shape as b(1,size of current layer)

    """

    A_prev, W, b = cache

    m = A_prev.shape[0] # Number of training examples

    

    dW = (1/m)*np.dot(A_prev.T,dZ) + 1./m*lambd*W # Derivative wrt Weights

    #print("dW",dW.shape)

    db = (1/m)*np.sum(dZ, axis=0, keepdims=True)  # Derivative wrt Bias

    #print("db",db.shape)

    dA_prev = np.dot(dZ,cache[1].T)

    

    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    

    return dA_prev, dW, db
def linear_activation_backward(dA, cache, lambd, activation):

    """

    This function implements backward propagation for LINEAR -> ACTIVATION layer

    

    @params



    dA = post-activation gradient for current layer l 

    cache = tuple of values (linear_cache, activation_cache) 

            we store for computing backward propagation efficiently

    activation = the activation to be used in this layer, 

                 stored as a text string: "sigmoid" or "relu"

    lambd = Regularization parameter, int

    

    Returns:



    dA_prev = Gradient of the cost with respect to the activation of the 

              previous layer l-1, 

              same shape as A_prev(number of examples, size of previous layer)

    dW = Gradient of the cost with respect to W of current layer l, 

         same shape as W(size of previous layer,size of current layer)

    db = Gradient of the cost with respect to b of current layer l, 

         same shape as b(1,size of current layer)

    """

    linear_cache, activation_cache = cache

    

    if activation == "relu":

        dZ = relu_backward(dA,activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

        

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA,activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    

    return dA_prev, dW, db
def L_model_backward(AL, Y, caches, lambd):

    """

    This function implements the backward propagation as following: 

    [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    

    @params



    AL = probability vector, output of the L_model_forward function

    Y = Ground Truth/ True "label" vector (containing classes 0 and 1) 

        shape  = (number of examples, 1)

    caches = list of caches containing:

             every cache of linear_activation_forward function with "relu" 

             (it's caches[l], for l in range(L-1) i.e l = 0...L-2)

             the cache of linear_activation_forward function with "sigmoid" 

             (it's caches[L-1])

    lambd = Regularization parameter, int

    

    Returns:



    grads = Dictionary with the gradients

            grads["dA" + str(l)] = ... 

            grads["dW" + str(l+1)] = ...

            grads["db" + str(l+1)] = ... 

    """

    grads = {}

    L = len(caches) # the number of layers

    m = AL.shape[0] # Number of training examples

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    

    # Initializing the backpropagation

    # Derivative of Binary Cross Entropy function

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    

    # Lth layer (SIGMOID -> LINEAR) gradients. 

    # Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, lambd, activation = "sigmoid")

    

    # Loop from l=L-2 to l=0

    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients

        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, lambd, activation = "relu")

        grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

        

    return grads
def initialize_adam(parameters) :

    """

    This function Initializes v and s as two python dictionaries with:

                - keys: "dW1", "db1", ..., "dWL", "dbL" 

                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters

    

    @param

    

    parameters = Dictionary containing parameters as follwoing:

                    parameters["W" + str(l)] = Wl

                    parameters["b" + str(l)] = bl

    

    Returns: 

    

    v = Dictionary that will contain the exponentially weighted average of the gradient

                    v["dW" + str(l)] = ...

                    v["db" + str(l)] = ...

    s = Dictionary that will contain the exponentially weighted average of the squared gradient

                    s["dW" + str(l)] = ...

                    s["db" + str(l)] = ...



    """

    

    L = len(parameters) // 2 # number of layers in the neural networks

    v = {}

    s = {}

    

    # Initialize v, s. Input: "parameters". Outputs: "v, s".

    for l in range(L):

        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)

        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

        s["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)

        s["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    

    return v, s
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,

                              beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

    """

    This function updates our model parameters using Adam



    @params

    

    parameters = Dictionary containing our parameters:

                  parameters['W' + str(l)] = Wl

                  parameters['b' + str(l)] = bl

    grads = Dictionary containing our gradients for each parameters:

                  grads['dW' + str(l)] = dWl

                  grads['db' + str(l)] = dbl

    v = Adam variable, moving average of the first gradient, python dictionary

    s = Adam variable, moving average of the squared gradient, python dictionary

    learning_rate = the learning rate, scalar.

    beta1 = Exponential decay hyperparameter for the first moment estimates 

    beta2 = Exponential decay hyperparameter for the second moment estimates 

    epsilon = hyperparameter preventing division by zero in Adam updates



    Returns:

    

    parameters = Dictionary containing our updated parameters 

    v = Adam variable, moving average of the first gradient, python dictionary

    s = Adam variable, moving average of the squared gradient, python dictionary

    """



    L = len(parameters) // 2                 # number of layers in the neural networks

    v_corrected = {}                         # Initializing first moment estimate, python dictionary

    s_corrected = {}                         # Initializing second moment estimate, python dictionary



    # Perform Adam update on all parameters

    for l in range(L):

        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".

        v["dW" + str(l+1)] = beta1 * v['dW' + str(l+1)] + (1 - beta1) * grads['dW' + str(l+1)]

        v["db" + str(l+1)] = beta1 * v['db' + str(l+1)] + (1 - beta1) * grads['db' + str(l+1)]



        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".

        v_corrected["dW" + str(l+1)] = v['dW' + str(l+1)] / float(1 - beta1**t)

        v_corrected["db" + str(l+1)] = v['db' + str(l+1)] / float(1 - beta1**t)



        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".

        s["dW" + str(l+1)] = beta2 * s['dW' + str(l+1)] + (1 - beta2) * (grads['dW' + str(l+1)]**2)

        s["db" + str(l+1)] = beta2 * s['db' + str(l+1)] + (1 - beta2) * (grads['db' + str(l+1)]**2)

          ### END CODE HERE ###



        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".  

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / float(1 - beta2**t)

        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / float(1 - beta2**t)



        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

    

    return parameters, v, s
def random_mini_batches(X, Y, mini_batch_size = 64):

    """

    This function creates a list of random minibatches from (X, Y)

    

    @params

    

    X = Data, numpy array of shape (number of examples, number of features)

    Y = Ground Truth/ True "label" vector (containing classes 0 and 1) 

        shape = (number of examples, 1)

    mini_batch_size = size of the mini-batches (suggested to use powers of 2)

    

    Returns:

    

    mini_batches = list of synchronous (mini_batch_X, mini_batch_Y)

    

    """

    

    np.random.seed(0)            

    m = X.shape[0]                  # Number of training examples

    mini_batches = []               # List to return synchronous minibatches

        

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:]

    #print("S_X",shuffled_X.shape)

    shuffled_Y = Y[permutation].reshape((m,1))

    #print("S_Y",shuffled_Y.shape)

    

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = (m//mini_batch_size) # number of mini batches of size mini_batch_size in our partitionning

    for k in range(num_complete_minibatches):

        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size,:]

        #print("M_X",mini_batch_X.shape)

        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)   # Tuple for synchronous minibatches

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size :,: ]

        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size :,: ]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, mini_batch_size = 128,n=1,lambd=0.7,

          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True): #lr was 0.009

    """

    This function implements a L-layer neural network: 

    [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

    

    Arguments:

    X = Data, numpy array of shape (number of examples, number of features)

    Y = Ground Truth/ True "label" vector (containing classes 0 and 1) 

        shape = (number of examples, 1)

    layers_dims = List contains number of neurons in one respective layer

                  and [len(layer_dims) - 1] gives L Layer Neural Network

    learning_rate = learning rate of the gradient descent update rule

    lambd = Regularization parameter, int

    num_epochs -- number of epochs

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """



    np.random.seed(1)

    costs = []                         # keep track of cost

    t = 0                              # Used in Adam

    #n_b = 10

    # Parameters initialization.

    parameters = initialize_parameters_deep(layers_dims,n)

    v, s = initialize_adam(parameters)

    

    # MiniBatch Gradient Descent

    for i in range(num_epochs):

        minibatches = random_mini_batches(X, Y, mini_batch_size=64)

        for minibatch in minibatches:

          # Select a minibatch

          (minibatch_X, minibatch_Y) = minibatch

          #print(minibatch_X.shape,i)

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

          AL, caches = L_model_forward(minibatch_X, parameters)

        

        # Compute cost

          cost = compute_cost(AL, minibatch_Y, parameters, lambd)

    

        # Backward propagation

          grads = L_model_backward(AL, minibatch_Y, caches,lambd)

 

        # Update parameters

          t += 1

          parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)

                

        # Print the cost every 100 training example

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" % (i, cost))

        if print_cost and i % 100 == 0:

            costs.append(cost)

            

    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title(("Learning rate = {}, Lambda = {} ".format(str(learning_rate),str(lambd))))

    plt.show()

    

    return parameters
def predict(X, y, parameters):

    """

    This function is used to predict the results of a  L-layer neural network

    

    @params



    X = Data, numpy array of shape (number of examples, number of features)

    parameters = Parameters of trained model returned by L_layer_model function 

    

    Returns:



    p = Predictions for the given dataset X

    """

    

    m = X.shape[0] # Number of training examples in Dataset

    n = len(parameters) // 2 # Number of layers in the neural network

    p = np.zeros((m,1))



    # Forward propagation

    probas, caches = L_model_forward(X, parameters)

    

    # Set values in p to 0/1 as per predictions and threshold

    for i in range(probas.shape[0]):

        # As per sigmoid, values greater than 0.5 are categorized as 1

        # and values lesser than 0.5 as categorized as 0

        if probas[i] > 0.5:

            p[i] = 1

        else:

            p[i] = 0

    #p = np.squeeze(p)

    y = y.reshape(p.shape)

    acc = np.sum((p == y)/m)*100

    print("Accuracy:%.2f%%" % acc)



    return p
learned_parameters = L_layer_model(X, y, layers_dims=[19,10,1],  mini_batch_size =128, n=1, learning_rate = 0.0001,lambd=0.01, num_epochs = 4000)
pred = predict(X,y,learned_parameters)

print(np.unique(pred), pred.shape)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y, pred))

print(confusion_matrix(y, pred))