import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.linear_model import LogisticRegression

import h5py

import sklearn.metrics



%matplotlib inline

np.random.seed(1)
# Helper Functions to Load the Datasets

def get_training_testing_data(train, test):

    # Loading the data files

    train_file = h5py.File(train, 'r')

    test_file = h5py.File(test, 'r')

    

    # Extracting the arrays from File object

    x_train = train_file['train_set_x'].value

    y_train = train_file['train_set_y'].value

    x_test = test_file['test_set_x'].value

    y_test = test_file['test_set_y'].value



    train_file.close()

    test_file.close()

    

    # Checking the dimensions and format of the data

    print('x_train.shape:', x_train.shape)

    print('y_train.shape:', y_train.shape)

    print('x_test.shape:', x_test.shape)

    print('y_test.shape:', y_test.shape)

    plt.imshow(x_train[0])

        

    # reshaping to convert X an Y to 2D array

#     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])

    y_train = y_train.reshape(y_train.shape[0], 1)

#     x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])

    y_test = y_test.reshape(y_test.shape[0], 1)



#     print('reshaped - ')

#     print('x_train.shape:', x_train.shape)

#     print('y_train.shape:', y_train.shape)

#     print('x_test.shape:', x_test.shape)

#     print('y_test.shape:', y_test.shape)

    

    # Normalising the values to floats between 0 to 1

    x_train = x_train/255

    x_test = x_test/255

    

    # Getting the values of m_train, m_test, and n

    m_train = x_train.shape[1]

    m_test = x_test.shape[1]

    n = x_train.shape[0]

    

    # Returning all the data extracted

    return x_train, y_train, x_test, y_test

    

# Creating or Loading the Data:

X, Y, X_test, Y_test =  get_training_testing_data('../input/train_catvnoncat.h5', '../input/test_catvnoncat.h5')



# Checking the shape of the Data

print(X.shape)

print(Y.shape)



# Sanity Check:

print('training sample:', X[0], Y[0])

print('testing sample: ',X_test[0], Y_test[0])
def relu(x):

    if x > 0:

        return x

    else:

        return 0



relu = np.vectorize(relu)



def max_pool(X):

    return X.max()
# Structure of the convolutional neural network:

convolutional_neural_network = {

    'dimensions' : {

        'input' : (64, 64, 3), 

        'filter' : (3, 3), 

        'conv' : None, 

        'activation' : None,

        'pooled' : None,

        'flattened' : None,

        'fc' : (1)

    }, 

    'number_of_channels' : { 

        'input' : 3, 

        'conv' : 2, 

        'fc' : 1

    }, 

    'stride' : 1, 

    'pad' : 1, 

    'pool' : 2,

    'pooling_function' : max_pool,

    'activation_function' : relu,

    'W' : {

        'conv' : None, 

        'fc' : None

    }, 

    'b' : {

        'conv' : None, 

        'fc' : None

    }

}  
def init_parameters(convolutional_neural_network):

    # Read only:

    number_of_channels = convolutional_neural_network['number_of_channels']

    pad = convolutional_neural_network['pad']

    pool = convolutional_neural_network['pool']

    stride = convolutional_neural_network['stride']

    

    # Read and write:

    dimensions = convolutional_neural_network['dimensions']

    W = convolutional_neural_network['W']

    b = convolutional_neural_network['b']



    # Defining the dimensions of the output of the Convolutional Layer, Pooling Layer and Flattening Layer:

    dimensions['conv'] = ( (dimensions['input'][0] - dimensions['filter'][0] + 2 * pad)//stride + 1, (dimensions['input'][1] - dimensions['filter'][1] + 2 * pad)//stride + 1, number_of_channels['conv'])  

    dimensions['activation'] = dimensions['conv']

    dimensions['pooled'] = (dimensions['conv'][0]//pool, dimensions['conv'][1]//pool, dimensions['conv'][2])

    dimensions['flattened'] = (dimensions['pooled'][0] * dimensions['pooled'][1] * dimensions['pooled'][2],)

    

    # Initialising the parameter W for both the convolutional layer and fully connected layer:

    W['conv'] = np.random.randn(dimensions['filter'][0], dimensions['filter'][1], number_of_channels['input'], number_of_channels['conv'])

    W['fc'] = np.random.randn(dimensions['flattened'][0], number_of_channels['fc'])



    # Initialising the parameter b for both the convolutional layer and fulley connected layer:

    b['conv'] = np.random.randn(number_of_channels['conv'])

    b['fc'] = np.random.randn(number_of_channels['fc'])

    

    # Write to the convolutional_neural_network to update it:

    convolutional_neural_network['dimensions'] = dimensions

    convolutional_neural_network['W'] = W

    convolutional_neural_network['b'] = b
def forward_propagation(A_input, convolutional_neural_network):

    # Read only:

    W = convolutional_neural_network['W']

    b = convolutional_neural_network['b']

    pool = convolutional_neural_network['pool']

    

    # Convolve over the input:

    A_conv = convolutional_layer(A_input, convolutional_neural_network)

    # Apply activation function over it:

    A_activation = activation_layer(A_conv, convolutional_neural_network)

    # Pool the output obtained:

    A_pooled = pooling_layer(A_activation, convolutional_neural_network)

    # Flatten the multi-dim to form a 2D matrix:

    A_flattened = flattening_layer(A_pooled, convolutional_neural_network)

    # Pass the flattened output through a fully connected layer:

    A_fc = fully_connected_layer(A_flattened, convolutional_neural_network)

    

    return A_fc
def convolutional_layer(A_input, convolutional_neural_network):

    # Read only:

    dimensions = convolutional_neural_network['dimensions']

    stride = convolutional_neural_network['stride']

    W = convolutional_neural_network['W']['conv']

    b = convolutional_neural_network['b']['conv']

    

    A_conv = np.zeros((A_input.shape[0] ,dimensions['conv'][0], dimensions['conv'][1], dimensions['conv'][2]))

    m = A_conv.shape[0]

    r = A_conv.shape[1]

    c = A_conv.shape[2]

    channels = A_conv.shape[3]

    

    # Process each window individually (Apply filter to each position one by one):

    for sample in range(m):

        for i in range(r):

            for j in range(c):

                for ch in range(channels):

                    A_conv[sample, i, j, ch] = (A_input[sample, i * stride: i * stride + dimensions['filter'][0], j * stride: j * stride + dimensions['filter'][1],:] * W[:, :, :, ch]).sum() + b[ch]     

    return A_conv
def activation_layer(A_conv, convolutional_neural_network):

    # Read only:

    activation_function = convolutional_neural_network['activation_function']

    

    # Apply activation function over A_conv

    A_activation = activation_function(A_conv)

    return A_activation
def pooling_layer(A_activation, convolutional_neural_network):

    # Read only:

    dimensions = convolutional_neural_network['dimensions']

    pool = convolutional_neural_network['pool']

    pooling_function = convolutional_neural_network['pooling_function']

    

    

    A_pooled = np.zeros((A_activation.shape[0], dimensions['pooled'][0], dimensions['pooled'][1], dimensions['pooled'][2]))

    m = A_pooled.shape[0]

    r = A_pooled.shape[1]

    c = A_pooled.shape[2]

    channels = A_pooled.shape[3]

    

    # Process each window individually

    for sample in range(m):

        for i in range(r):

            for j in range(c):

                for ch in range(channels):

                    A_pooled[sample, i, j, ch] = pooling_function(A_activation[sample, i * pool : i * pool + pool, j * pool : j * pool + pool, ch])   

    return A_pooled
def flattening_layer(A_pooled, convolutional_neural_network):

    # Read only:

    dimensions = convolutional_neural_network['dimensions']

    

    A_flattened = np.reshape(A_pooled, (A_pooled.shape[0], dimensions['flattened'][0]))

    return A_flattened
def fully_connected_layer(A_flattened, convolutional_neural_network):

    # Read only:

    W = convolutional_neural_network['W']

    b = convolutional_neural_network['b']

    

    A_fc = (A_flattened @ W['fc'] + b['fc'])

    

    return A_fc
def backward_propagation(A, Y, convolutional_neural_network):

    # Coming Soon

    pass
def accuracy(A, Y):

    return (A.T @ Y)[0][0] / A.shape[0]
def zero_pad(x, pad):

    # Add padding to the image:

    x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values = (0, 0))

    return x_padded
def fit(X, Y, convolutional_neural_network,epochs = 1, learning_rate = 0.01):

    # Ensure the input given is correct:

    assert convolutional_neural_network['number_of_channels']['input'] == X.shape[3]

    assert convolutional_neural_network['number_of_channels']['fc'] == Y.shape[1]

    assert convolutional_neural_network['dimensions']['input'] == X.shape[1:]

    

    # Pad the input data:

    pad = convolutional_neural_network['pad']

    A_input = zero_pad(X, pad)

    

    # Initialise all the parameters of the convolutional neural network:

    init_parameters(convolutional_neural_network)



    # Train the model:

    for epoch in range(1, epochs + 1):

        A = forward_propagation(A_input, convolutional_neural_network)

        acc = accuracy(A, Y)

        print('epoch : ',epoch, '-->', acc)

        backward_propagation(A, Y, convolutional_neural_network)

fit(X, Y, convolutional_neural_network)