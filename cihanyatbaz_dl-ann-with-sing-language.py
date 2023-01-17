# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode('utf8'))

# Any results you write to the current directory are saved as output.
# Load datasets
x_data = np.load('../input/Sign-language-digits-dataset/X.npy')
y_data = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64  # pixel size

 # for sign zero
plt.subplot(1,2,1)  
plt.imshow(x_data[260])  # Get 260th index
plt.axis("off")

# for sign one
plt.subplot(1,2,2)
plt.imshow(x_data[900])  # Get 900th index
plt.axis("off")
# From 0 to 204 is zero sign, from 205 to 410 is one sign
X = np.concatenate((x_data[204:409], x_data[822:1027] ), axis=0)

# We will create their labels. After that, we will concatenate on the Y.
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z,o), axis=0).reshape(X.shape[0],1)
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=42)
# random_state = Use same seed while randomizing
print(x_train.shape)
print(y_train.shape)
x_train_flatten = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test_flatten = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print('x_train_flatten: {} \nx_test_flatten: {} '.format(x_train_flatten.shape, x_test_flatten.shape))
# Here we will change the location of our samples and features. '(328,4096) -> (4096,328)' 
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_train = y_train.T
y_test = y_test.T

print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)
# Now let's create the parameter and sigmoid function. 
# So what we need is dimension 4096 that is number of pixel as a parameter for our initialize method(def)

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w,b

# Sigmoid function
# z = np.dot(w.T, x_train) +b
def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))  # sigmoid function finding formula
    return y_head
sigmoid(0)  # 0 should result in 0.5
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {
        "weight1": np.random.randn(3,x_train.shape[0]) * 0.1, 
# the reason we say 3, 4096: The number of rows in our weight has to be 3 because we have 3 nodes
        "bias1": np.zeros((3,1)), # That is same for bias 
        "weight2": np.random.randn(y_train.shape[0], 3) * 0.1,
        "bias2": np.zeros((y_train.shape[0],1))
    }
    return parameters
# We only process 2 times because we use tanh function.
def forward_NN(x_train, parameters):
    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)  # We can do this easily with the numpy library
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache
# Compute cost
# We take A2 as input and use it here
def compute_cost_NN(A2, Y, parameters):
# We multiply the y_head value (A2) with the actual single value (Y).
    logprobs = np.multiply(np.log(A2),Y)
# cost : We collect all our losts
    cost = - np.sum(logprobs) / Y.shape[1]
    return cost
def backward_NN(parameters, cache, X, Y):
    # We are doing derivative transactions
    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T) / X.shape[1]
    # keepdims : He's holding as an Array. We are writing in array, even though the result of our collection is different (consten).
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1, X.T) / X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]
    # grads : We're storing grads. Changes in Weight1, bias1, Weight2, bias2. We store the derivatives of these according to them.
    grads = {
        "dweight1": dW1,
        "dbias1": db1,
        "dweight2": dW2,
        "dbias2": db2,
    }
    return grads
# Learning_rate: It is a hyper parameter. A parameter we'll find by trying.
def update_NN(parameters, grads, learning_rate = 0.003):
    parameters = {
        "weight1": parameters["weight1"] - learning_rate * grads["dweight1"],
        "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],
        "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],
        "bias2": parameters["bias2"] - learning_rate * grads["dbias2"],
    }
    return parameters
def predict_NN(parameters, x_test):
    #x_test is a input for forward propagation
    A2, cache = forward_NN(x_test, parameters)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our predictioan is sign one(y_head=1),
    # if z is smaller than 0.5, our predictioan is sign zero(y_head=0),    
    for i in range(A2.shape[1]):
        if A2[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    return Y_prediction
# Layer Neural Network
def layer_neural_network(x_train, y_train, x_test, y_test, num_iterations):
    # We store Cost and Indexes for analysis.
    cost_list = []
    index_list = []
    # initialize parameters and layer sizes
    # We determine how many nodes in our layer.
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)
    
    for i in range(0, num_iterations):
        # forward propagation
        A2, cache = forward_NN(x_train, parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
        # backward propagation
        grads  = backward_NN(parameters, cache, x_train, y_train)
        # update parameters
        parameters = update_NN(parameters, grads)
        
        # It will store cost in cost_list per hundred steps. Same goes for index.
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
    
    plt.plot(index_list, cost_list)
    plt.xticks(index_list, rotation='vertical')
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict_NN(parameters, x_test)
    y_prediction_train = predict_NN(parameters, x_train)
    
    # Print test/train errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))    

    return parameters

parameters = layer_neural_network(x_train, y_train, x_test, y_test, num_iterations = 2500) 
# We need to get transposing when using Keras. Because it's easier.
# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T  
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
# build classifier: We are building a structure that will form the neural network.
def build_classifier():
    classifier = Sequential()  # initialize neural network
    
    # Dense: It's building the layers.
    #   - units=8: We have eight node.
    #   - kernel_initializer: The values we first define in weights. It will be randomly distributed with uniform.  
    #   - relu: If input < 0, x = 0 indicates(sample: relu(-6)=0). If input > 0, x = x indicates(sample: relu(6)=6).
    #   - input_dim: 4096 px.
    classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    # - sigmoid: We're adding our last output layer. That will be our output layer.
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))            
    
    # We will find loss and cost.
    # - adam: Adaptive momentum. If we use Adam, learning_rate is not fixed. It updates Learning_rate and enables us to learn more quickly.
    # - loss: The same lost function that we use in Linear Regression.
    # - metrics: Evaluation metric. We choose Accuracy.
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# - build_fn: This parameter calls the neural network we built.
# epochs: number of iteration
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
    
# cross_val_score: It gives us more than one accuracy and we get a more effective result by taking averages of them. 
# estimator: We determine the classifier to use.
# cv : Find 3 times accuracy and then we'll average it, after that we find a more effective and more accurate result.
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv =3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean:" + str(mean))
print("Accuracy variance:" + str(variance))