import numpy as np

np.random.seed(0)



def sigmoid(x): # Returns values that sums to one.

    return 1 / (1 + np.exp(-x))



def sigmoid_derivative(sx):

    # See https://math.stackexchange.com/a/1225116

    return sx * (1 - sx)



# Cost functions.

def cost(predicted, truth):

    return truth - predicted



xor_input = np.array([[0,0], [0,1], [1,0], [1,1]])

xor_output = np.array([[0,1,1,0]]).T



# Lets drop the last row of data and use that as unseen test.

X = xor_input

Y = xor_output



# Define the shape of the weight vector.

num_data, input_dim = X.shape

# Lets set the dimensions for the intermediate layer.

hidden_dim = 5

# Initialize weights between the input layers and the hidden layer.

W1 = np.random.random((input_dim, hidden_dim))



# Define the shape of the output vector. 

output_dim = len(Y.T)

# Initialize weights between the hidden layers and the output layer.

W2 = np.random.random((hidden_dim, output_dim))



num_epochs = 10000

learning_rate = 1.0



for epoch_n in range(num_epochs):

    layer0 = X

    # Forward propagation.

    

    # Inside the perceptron, Step 2. 

    layer1 = sigmoid(np.dot(layer0, W1))

    layer2 = sigmoid(np.dot(layer1, W2))



    # Back propagation (Y -> layer2)

    

    # How much did we miss in the predictions?

    layer2_error = cost(layer2, Y)

    # In what direction is the target value?

    # Were we really close? If so, don't change too much.

    layer2_delta = layer2_error * sigmoid_derivative(layer2)



    

    # Back propagation (layer2 -> layer1)

    # How much did each layer1 value contribute to the layer2 error (according to the weights)?

    layer1_error = np.dot(layer2_delta, W2.T)

    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    

    # update weights

    W2 +=  learning_rate * np.dot(layer1.T, layer2_delta)

    W1 +=  learning_rate * np.dot(layer0.T, layer1_delta)
for x, y in zip(X, Y):

    layer1_prediction = sigmoid(np.dot(W1.T, x)) # Feed the unseen input into trained W.

    prediction = layer2_prediction = sigmoid(np.dot(W2.T, layer1_prediction)) # Feed the unseen input into trained W.

    print(int(prediction > 0.5), y)
import numpy as np

np.random.seed(0)



def sigmoid(x): # Returns values that sums to one.

    return 1 / (1 + np.exp(-x))



def sigmoid_derivative(sx):

    # See https://math.stackexchange.com/a/1225116

    return sx * (1 - sx)



# Cost functions.

def cost(predicted, truth):

    return truth - predicted



xor_input = np.array([[0,0], [0,1], [1,0], [1,1]])

xor_output = np.array([[0,1,1,0]]).T



# Lets drop the last row of data and use that as unseen test.

X = xor_input[:-1]

Y = xor_output[:-1]



# Define the shape of the weight vector.

num_data, input_dim = X.shape

# Lets set the dimensions for the intermediate layer.

hidden_dim = 5

# Initialize weights between the input layers and the hidden layer.

W1 = np.random.random((input_dim, hidden_dim))



# Define the shape of the output vector. 

output_dim = len(Y.T)

# Initialize weights between the hidden layers and the output layer.

W2 = np.random.random((hidden_dim, output_dim))



num_epochs = 10000

learning_rate = 1.0



for epoch_n in range(num_epochs):

    layer0 = X

    # Forward propagation.

    

    # Inside the perceptron, Step 2. 

    layer1 = sigmoid(np.dot(layer0, W1))

    layer2 = sigmoid(np.dot(layer1, W2))



    # Back propagation (Y -> layer2)

    

    # How much did we miss in the predictions?

    layer2_error = cost(layer2, Y)

    # In what direction is the target value?

    # Were we really close? If so, don't change too much.

    layer2_delta = layer2_error * sigmoid_derivative(layer2)



    

    # Back propagation (layer2 -> layer1)

    # How much did each layer1 value contribute to the layer2 error (according to the weights)?

    layer1_error = np.dot(layer2_delta, W2.T)

    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    

    # update weights

    W2 +=  learning_rate * np.dot(layer1.T, layer2_delta)

    W1 +=  learning_rate * np.dot(layer0.T, layer1_delta)
for x, y in zip(xor_input, xor_output):

    layer1_prediction = sigmoid(np.dot(W1.T, x)) # Feed the unseen input into trained W.

    prediction = layer2_prediction = sigmoid(np.dot(W2.T, layer1_prediction)) # Feed the unseen input into trained W.

    print(int(prediction > 0.5), y)
import numpy as np

import random

np.random.seed(0)



def sigmoid(x): # Returns values that sums to one.

    return 1 / (1 + np.exp(-x))



def sigmoid_derivative(sx):

    # See https://math.stackexchange.com/a/1225116

    return sx * (1 - sx)



# Cost functions.

def cost(predicted, truth):

    return truth - predicted



X = xor_input = np.array([[0,0], [0,1], [1,0], [1,1]])

Y = xor_output = np.array([[0,1,1,0]]).T



# Shuffle the order of the inputs

_temp = list(zip(X, Y))

random.shuffle(_temp)

xor_input_shuff, xor_output_shuff = map(np.array, zip(*_temp))



# Lets drop the last row of data and use that as unseen test.

X = xor_input_shuff[:-1]

Y = xor_output_shuff[:-1]



# Define the shape of the weight vector.

num_data, input_dim = X.shape

# Lets set the dimensions for the intermediate layer.

hidden_dim = 5

# Initialize weights between the input layers and the hidden layer.

W1 = np.random.random((input_dim, hidden_dim))



# Define the shape of the output vector. 

output_dim = len(Y.T)

# Initialize weights between the hidden layers and the output layer.

W2 = np.random.random((hidden_dim, output_dim))



num_epochs = 10000

learning_rate = 1.0



for epoch_n in range(num_epochs):

    layer0 = X

    # Forward propagation.

    

    # Inside the perceptron, Step 2. 

    layer1 = sigmoid(np.dot(layer0, W1))

    layer2 = sigmoid(np.dot(layer1, W2))



    # Back propagation (Y -> layer2)

    

    # How much did we miss in the predictions?

    layer2_error = cost(layer2, Y)

    # In what direction is the target value?

    # Were we really close? If so, don't change too much.

    layer2_delta = layer2_error * sigmoid_derivative(layer2)



    

    # Back propagation (layer2 -> layer1)

    # How much did each layer1 value contribute to the layer2 error (according to the weights)?

    layer1_error = np.dot(layer2_delta, W2.T)

    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    

    # update weights

    W2 +=  learning_rate * np.dot(layer1.T, layer2_delta)

    W1 +=  learning_rate * np.dot(layer0.T, layer1_delta)
for x, y in zip(xor_input, xor_output):

    layer1_prediction = sigmoid(np.dot(W1.T, x)) # Feed the unseen input into trained W.

    prediction = layer2_prediction = sigmoid(np.dot(W2.T, layer1_prediction)) # Feed the unseen input into trained W.

    print(x, int(prediction > 0.5), y)
import random

random.seed(0)



def generate_zero():

    return random.uniform(0, 49) / 100



def generate_one():

    return random.uniform(50, 100) / 100





def generate_xor_XY(num_data_points):

    Xs, Ys = [], []

    for _ in range(num_data_points):

        # xor(0, 0) -> 0 

        Xs.append([generate_zero(), generate_zero()]); Ys.append([0])

        # xor(1, 0) -> 1

        Xs.append([generate_one(), generate_zero()]); Ys.append([1])

        # xor(0, 1) -> 1

        Xs.append([generate_zero(), generate_one()]); Ys.append([1])

        # xor(1, 1) -> 0

        Xs.append([generate_one(), generate_one()]); Ys.append([0])

    return Xs, Ys

X, Y = generate_xor_XY(100)

X = np.array(X)

Y = np.array(Y)
# First 20 instance of new data.

for i, (x, y) in enumerate(zip(X, Y)):

    if i > 20:

        break

    print(x, [int(_x > 0.5) for _x in x],  y)

    
import numpy as np

import random

np.random.seed(0)



def sigmoid(x): # Returns values that sums to one.

    return 1 / (1 + np.exp(-x))



def sigmoid_derivative(sx):

    # See https://math.stackexchange.com/a/1225116

    return sx * (1 - sx)



# Cost functions.

def cost(predicted, truth):

    return truth - predicted



# Shuffle the order of the inputs

_temp = list(zip(X, Y))

random.shuffle(_temp)

xor_input_shuff, xor_output_shuff = map(np.array, zip(*_temp))



# Lets split the data to 90-10. 

train_split = int(len(X) / 100 * 90)

X_train = xor_input_shuff[:train_split]

Y_train = xor_output_shuff[:train_split]



X_test = xor_input_shuff[train_split:]

Y_test = xor_output_shuff[train_split:]



# Define the shape of the weight vector.

num_data, input_dim = X_train.shape

# Lets set the dimensions for the intermediate layer.

hidden_dim = 5

# Initialize weights between the input layers and the hidden layer.

W1 = np.random.random((input_dim, hidden_dim))



# Define the shape of the output vector. 

output_dim = len(Y_train.T)

# Initialize weights between the hidden layers and the output layer.

W2 = np.random.random((hidden_dim, output_dim))



num_epochs = 2000

learning_rate = 0.03



for epoch_n in range(num_epochs):

    layer0 = X_train

    # Forward propagation.

    

    # Inside the perceptron, Step 2. 

    layer1 = sigmoid(np.dot(layer0, W1))

    layer2 = sigmoid(np.dot(layer1, W2))



    # Back propagation (Y -> layer2)

    

    # How much did we miss in the predictions?

    layer2_error = cost(layer2, Y_train)

    # In what direction is the target value?

    # Were we really close? If so, don't change too much.

    layer2_delta = layer2_error * sigmoid_derivative(layer2)



    

    # Back propagation (layer2 -> layer1)

    # How much did each layer1 value contribute to the layer2 error (according to the weights)?

    layer1_error = np.dot(layer2_delta, W2.T)

    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    

    # update weights

    W2 +=  learning_rate * np.dot(layer1.T, layer2_delta)

    W1 +=  learning_rate * np.dot(layer0.T, layer1_delta)
accurate = 0

for x, y in zip(X_test, Y_test):

    layer1_prediction = sigmoid(np.dot(W1.T, x)) # Feed the unseen input into trained W.

    prediction = layer2_prediction = sigmoid(np.dot(W2.T, layer1_prediction)) # Feed the unseen input into trained W.

    print(x, [int(_x > 0.5) for _x in x], int(prediction > 0.5), y)

    accurate += int(prediction > 0.5) == y
print(accurate/len(X_test))