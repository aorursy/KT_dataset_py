# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

# filter warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x = np.load('../input/sign-language-digits-dataset/X.npy')

y = np.load('../input/sign-language-digits-dataset/Y.npy')



img_size = 64



plt.subplot(1,2,1)

plt.imshow(x[260].reshape( img_size , img_size))

plt.subplot(1,2,2)

plt.imshow(x[900].reshape( img_size , img_size))



plt.show()
X = np.concatenate( (x[204:409] , x[822:1027]) , axis = 0 )



zeros = np.zeros( 205 )

ones  = np.ones( 205 )



Y = np.concatenate( (zeros , ones) , axis = 0 ).reshape(X.shape[0],1)



print("X shape: " , X.shape)

print("Y shape: " , Y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.15, random_state = 42)



X_train_size = X_train.shape[0]

X_test_size  = X_test.shape[0]



print("X Train Size: " , X_train_size)

print("X Test  Size: " , X_test_size)
X_train_flattened = X_train.reshape( X_train_size, X_train.shape[1] * X_train.shape[2] )

X_test_flattened  = X_test.reshape ( X_test_size, X_test.shape[1] * X_test.shape[2] )



print("X train flattened",X_train_flattened.shape)

print("X test flattened ",X_test_flattened.shape)
x_train = X_train_flattened.T

x_test  = X_test_flattened.T



y_train = Y_train.T

y_test  = Y_test.T



print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b
w , b = initialize_weights_and_bias( 4096 )

w.shape
def sigmoid( x ):

    return 1/ (1 + np.exp(-x))
def forwardPropagation( w, b, x_train, y_train ):

    z = np.dot( w.T , x_train ) # result will be 348 x 1

    

    y_head = sigmoid( z )       # Between 0 - 1

    

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    return cost 
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation part

    

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    

    # backward propagation part

    

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iteration):

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
def predict( w, b, x_test ):

    

    z = np.dot( w.T, x_test) + b

    

    s = sigmoid( z )

    

    predictions = np.zeros( (1,x_test.shape[1]))

    

    for i in range( s.shape[1] ):

        if( s[0,i] > 0.5 ):

            predictions[0,i] = 1

        else:

            predictions[0,i] = 0

    

    return predictions
def logisticRegression( x_train, x_test, y_train, y_test, learning_rate, num_of_iteration):

    

    dimension = x_train.shape[0]

    

    w, b = initialize_weights_and_bias( dimension )

    

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_of_iteration)

    

    y_train_predictions = predict( parameters['weight'], parameters['bias'], x_train )

    

    y_test_predictions  = predict( parameters['weight'], parameters['bias'], x_test )

    

    print('Train Accuracy: {} %'.format(  100 - 100 * ( np.mean(np.abs(y_train - y_train_predictions))) ) )

    

    print('Test Accuracy: {} %'.format(  100 - 100 * ( np.mean(np.abs(y_test - y_test_predictions))) ) ) 
logisticRegression( x_train, x_test, y_train, y_test, learning_rate = 0.01, num_of_iteration = 150)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42, max_iter= 1000)



model = logreg.fit( x_train.T, y_train.T )

train_score = model.score( x_test.T, y_test.T)



print('Test Score : {} '.format(train_score))
x_train.shape[0]
def initializeParamsAndLayerSize(  x_train, y_train ):

    

    params = {

        

        'weight1' : np.random.randn(3,x_train.shape[0]) * 0.1,

        'bias1'   : np.zeros( (3,1) ),

        

        'weight2' : np.random.randn(y_train.shape[0],3) * 0.1,

        'bias2'   : np.zeros((y_train.shape[0],1))

    }

    return params

    
def forward_propagation_NN( x_train , params ):

    

    z1 = np.dot( params['weight1'] , x_train ) + params['bias1']

    a1 = np.tanh( z1 )

    

    z2 = np.dot( params['weight2'] , a1 ) + params['bias2']

    a2 = sigmoid( z2 )

      

    cache = {

        "Z1": z1,

        "A1": a1,

        "Z2": z2,

        "A2": a2

    }

    

    return a2, cache
def compute_cost_NN(A2, Y):

    

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    return cost
# Backward Propagation

def backward_propagation_NN(parameters, cache, X, Y):



    dZ2 = cache["A2"]-Y

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads
def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"] - learning_rate * grads["dweight1"],

                  "bias1"  : parameters["bias1"]   - learning_rate * grads["dbias1"],

                  "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],

                  "bias2"  : parameters["bias2"]   - learning_rate * grads["dbias2"]}

    

    return parameters
def predict( params, x_test ):

    

    A2, cache = forward_propagation_NN( x_test, params )

    

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(A2.shape[1]):

        

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    

    return Y_prediction
def two_layer_network( x_train, x_test, y_train, y_test, no_of_iter):

    

    cost_list  = []

    index_list = []

    

    

    parameters = initializeParamsAndLayerSize(x_train, y_train)

    

    for i in range( 0, no_of_iter):

        y_head, cache = forward_propagation_NN( x_train, parameters )

        

  

        cost      = compute_cost_NN( y_head, y_train )

        

        gradients = backward_propagation_NN( parameters, cache, x_train, y_train )

        

        parameters    = update_parameters_NN(parameters, gradients)

        

        if i % 100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))



    

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    

    y_prediction_test  = predict(parameters,x_test)

    y_prediction_train = predict(parameters,x_train)



    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

    return 



two_layer_network( x_train, x_test, y_train, y_test, 1000)
# FOR EASE OF USE IN KERAS



x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library
def buildClassifier():

    

    classifier = Sequential() # initialize neural network



    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))



    classifier.add( Dense( units = 4, kernel_initializer = 'uniform', activation = 'relu'))



    classifier.add( Dense( units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    

    return classifier
classifier = KerasClassifier(build_fn = buildClassifier, epochs = 100)



accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))