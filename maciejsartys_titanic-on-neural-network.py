# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# helper functions for deep neural network model

def sigmoid(Z):

    """

    Implements the sigmoid activation in numpy

    

    Arguments:

    Z -- numpy array of any shape

    

    Returns:

    A -- output of sigmoid(z), same shape as Z

    cache -- returns Z as well, useful during backpropagation

    """

    A = 1 / ( 1 + np.exp(-Z) )

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



def initialize_parameters_deep(layer_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the dimensions of each layer in our network

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])

                    bl -- bias vector of shape (layer_dims[l], 1)

    """

    parameters = {}

    L = len(layer_dims)            # number of layers in the network



    for l in range(1, L):

        ### START CODE HERE ### (≈ 2 lines of code)

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        ### END CODE HERE ###

        

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))

        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))



        

    return parameters



def linear_forward(A, W, b):

    """

    Implement the linear part of a layer's forward propagation.



    Arguments:

    A -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)



    Returns:

    Z -- the input of the activation function, also called pre-activation parameter 

    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """

    

    ### START CODE HERE ### (≈ 1 line of code)

    Z = np.dot(W, A) + b

    ### END CODE HERE ###

    

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

    cache -- a python dictionary containing "linear_cache" and "activation_cache";

             stored for computing the backward pass efficiently

    """

    

    if activation == "sigmoid":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W , b)

        Z = np.float64(Z)

        A, activation_cache = sigmoid(Z)

    

    elif activation == "relu":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W , b)

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

                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)

                the cache of linear_sigmoid_forward() (there is one, indexed L-1)

    """



    caches = []

    A = X

    L = len(parameters) // 2                  # number of layers in the neural network

    

    for l in range(1, L):

        A_prev = A 

        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], 'relu')

        caches.append(cache)

        

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], 'sigmoid')

    caches.append(cache)

    

    assert(AL.shape == (1,X.shape[1]))

            

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

    cost = -np.sum(np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1-AL).T))/m

    

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    assert(cost.shape == ())

    

    return cost



def linear_backward(dZ, cache):

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



    dW = np.dot(dZ, A_prev.T) / m

    db = np.sum(dZ, axis = 1, keepdims = True) / m

    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    

    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):

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

        dZ = relu_backward(dA, cache[1])

        dA_prev, dW, db = linear_backward(dZ, cache[0])

        

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, cache[1])

        dA_prev, dW, db = linear_backward(dZ, cache[0])

    

    return dA_prev, dW, db





def L_model_backward(AL, Y, caches):

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

    

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]

    current_cache = caches[L-1]

    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    

    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients.

        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, 'relu')

        grads["dA" + str(l + 1)] = dA_prev_temp

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

    for i in range(1, L+1):

        parameters["W" + str(i)] += - learning_rate * grads["dW" + str(i)]

        parameters["b" + str(i)] += - learning_rate * grads["db" + str(i)]

    return parameters



def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009

    """

    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    

    Arguments:

    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).

    learning_rate -- learning rate of the gradient descent update rule

    num_iterations -- number of iterations of the optimization loop

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """



    np.random.seed(123)

    costs = []                         # keep track of cost

    

    # Parameters initialization.

    parameters = initialize_parameters_deep(layers_dims)

    

    # Loop (gradient descent)

    for i in range(0, num_iterations):



        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

        AL, caches = L_model_forward(X, parameters)

        

        # Compute cost.

        #print(AL)

        cost = compute_cost(AL, Y)

    

        # Backward propagation.

        grads = L_model_backward(AL, Y, caches)

 

        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)

                

        # Print the cost every 10000 training example

        if print_cost and i % 10000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 10000 == 0:

            costs.append(cost)

            

    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
train = pd.read_csv('../input/train.csv', index_col = 0)

test_challange = pd.read_csv('../input/test.csv', index_col=0)

train.describe(include = 'all')

import missingno as msno

msno.bar(df=train)
def parseSex(df):

    df.loc[df['Sex'] == 'male', 'Sex'] = 1

    df.loc[df['Sex'] == 'female', 'Sex'] = 0

    return df

def addFamilySize(df):

    df['FamilySize'] = 0

    df['FamilySize'] = df['Parch'] + df['SibSp']

    return df

def parseEmbarked(df):

    df.Embarked = df.Embarked.replace({

        'S': 1.0,

        'C': 2.0,

        'Q': 3.0

    })

    return df
def normalize(series):

    mean = series.mean()

    stdev= series.std()

    return (series - mean)/stdev
def preprocess_data(df):

    df = df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

    df = parseSex(df)

    df = addFamilySize(df)

    df = df.drop(['SibSp', 'Parch'], axis = 1)

    df = parseEmbarked(df)

    embarked_median = df['Embarked'].median()

    df.Embarked = df.Embarked.fillna(embarked_median)

    age_median = df['Age'].median()

    df.Age = df.Age.fillna(age_median)

    df.Age = normalize(df.Age)

    df.FamilySize = normalize(df.FamilySize)

    #df.Pclass = normalize(df.Pclass)

    return df
def draw_factorplot(data, x):

    Pclass = sns.factorplot(x = x, y = 'Survived', data = data, kind = 'bar', palette = 'muted')

    Pclass.despine(left = True)
sex = draw_factorplot(train, 'Sex')

Pclass = draw_factorplot(train, 'Pclass')

embarked = draw_factorplot(train, 'Embarked')

Parch = draw_factorplot(train, 'Parch')

SibSp = draw_factorplot(train, 'SibSp')
train = preprocess_data(train)

train, test = train_test_split(train, test_size = 0.2)

predict_data = preprocess_data(test_challange)
features_list = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked']

train_features = np.float64(train[features_list].values)

train_target = np.float64(train['Survived'].values)

test_features = np.float64(test[features_list].values)

test_target = np.float64(test.Survived.values)

predict_data_features = np.float64(predict_data[features_list].values)
layers_dims = [6, 6, 4, 1]

features_t = train_features.T

target_t = train_target.T.reshape(1, train_target.shape[0])



test_features_t = test_features.T

test_target_t = test_target.T.reshape(1, test_target.shape[0])

predict_data_features_t = predict_data_features.T

def predict(X, y, parameters):

    """

    This function is used to predict the results of a  L-layer neural network.

    

    Arguments:

    X -- data set of examples you would like to label

    parameters -- parameters of the trained model

    

    Returns:

    p -- predictions for the given dataset X

    """

    

    m = X.shape[1]

    n = len(parameters) // 2 # number of layers in the neural network

    p = np.zeros((1,m))

    

    # Forward propagation

    probas, caches = L_model_forward(X, parameters)



    

    # convert probas to 0/1 predictions

    for i in range(0, probas.shape[1]):

        if probas[0,i] > 0.5:

            p[0,i] = 1

        else:

            p[0,i] = 0

    

    #print results

    #print ("predictions: " + str(p))

    #print ("true labels: " + str(y))

    print("Accuracy: "  + str(np.sum((p == y)/m)))

        

    return p
parameters = L_layer_model(features_t, target_t, layers_dims, learning_rate = 0.075, num_iterations = 150000, print_cost = True)
predict(features_t, target_t, parameters)

predict(test_features_t, test_target_t, parameters)
zeros = np.zeros(predict_data_features_t.shape)

prediction = predict(predict_data_features_t, zeros, parameters)
prediction.shape
ids = predict_data.index.values

final = pd.DataFrame({

    'PassengerId': ids,

    'Survived': prediction.reshape(418, ).astype('int')

})

final
final.to_csv('nn_sub.csv', index=False)