import numpy as np

import pandas as pd
class Layer:

    def __init__(self):

        self.input = None

        self.output = None



    # computes the output Y of a layer for a given input X

    def forward_propagation(self, input):

        raise NotImplementedError



    # computes dE/dX for a given dE/dY (and update parameters if any)

    def backward_propagation(self, output_error, learning_rate):

        raise NotImplementedError
# inherit from base class Layer

class FCLayer(Layer):

    # input_size = number of input neurons

    # output_size = number of output neurons

    def __init__(self, input_size, output_size):

        self.weights = np.random.rand(input_size, output_size) - 0.5

        self.bias = np.random.rand(1, output_size) - 0.5



    # returns output for a given input

    def forward_propagation(self, input_data):

        self.input = input_data

        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output



    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.

    def backward_propagation(self, output_error, learning_rate):

        input_error = np.dot(output_error, self.weights.T)

        weights_error = np.dot(self.input.T, output_error)

        # dBias = output_error



        # update parameters

        self.weights -= learning_rate * weights_error

        self.bias -= learning_rate * output_error

        return input_error
#loss function and its derivative

def mse(y_true, y_pred):

    return(np.mean(np.power(y_true-y_pred, 2)));



def mse_prime(y_true, y_pred):

    return(2*(y_pred-y_true)/y_true.size);
#activation function and derivative



def tanh(x):

    return(np.tanh(x));



def tanh_prime(x):

    return(1-np.tanh(x)**2);
#inherit from base class

class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime):

        self.activation = activation

        self.activation_prime = activation_prime

        

    #return the activation input

    def forward_propagation(self, input_data):

        self.input = input_data

        self.output = self.activation(self.input)

        return(self.output)

    

    #return input_error = dE/dX for a given output_error=dE/dY

    def backward_propagation(self, output_error, learning_rate):

        return(self.activation_prime(self.input) * output_error)
class Network:

    def __init__(self):

        self.layers = []

        self.loss = None

        self.loss_prime = None



    # add layer to network

    def add(self, layer):

        self.layers.append(layer)



    # set loss to use

    def use(self, loss, loss_prime):

        self.loss = loss

        self.loss_prime = loss_prime



    # predict output for given input

    def predict(self, input_data):

        # sample dimension first

        samples = len(input_data)

        result = []



        # run network over all samples

        for i in range(samples):

            # forward propagation

            output = input_data[i]

            for layer in self.layers:

                output = layer.forward_propagation(output)

            result.append(output)



        return result



    # train the network

    def fit(self, x_train, y_train, epochs, learning_rate):

        # sample dimension first

        samples = len(x_train)

        

        #saving epoch and error in list

        epoch_list = []

        error_list = []



        # training loop

        for i in range(epochs):

            err = 0

            for j in range(samples):

                # forward propagation

                output = x_train[j]

                for layer in self.layers:

                    output = layer.forward_propagation(output)



                # compute loss (for display purpose only)

                err += self.loss(y_train[j], output)



                # backward propagation

                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):

                    error = layer.backward_propagation(error, learning_rate)



            # calculate average error on all samples

            err /= samples

            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

            

            epoch_list.append(i+1)

            error_list.append(err)

        

            #creating dataframe of epoch and error

            df = pd.DataFrame()

            df['epoch'] = epoch_list

            df['loss'] = error_list

        return df
from keras.datasets import mnist

from keras.utils import np_utils
#load MNIST from server

(x_train, y_train),(x_test, y_test) = mnist.load_data()



#training data

#reshape and normalize input data

x_train = x_train.reshape(x_train.shape[0],1,28*28)

x_train = x_train.astype('float32')

x_train /= 255



#ecnoding output

y_train = np_utils.to_categorical(y_train)



#same for test data

x_test = x_test.reshape(x_test.shape[0],1,28*28)

x_test = x_test.astype('float32')

x_test /= 255



#ecnoding output

y_test = np_utils.to_categorical(y_test)
#Network

net = Network()

net.add(FCLayer(28*28, 50))

net.add(ActivationLayer(tanh, tanh_prime))

net.add(FCLayer(50, 10))

net.add(ActivationLayer(tanh, tanh_prime))



net.use(mse, mse_prime)
df = net.fit(x_train[:1000], y_train[:1000],epochs=50,learning_rate=0.1)
import plotly.express as px



fig = px.line(df, x='epoch', y='loss',title='Change in loss with respect to Epochs')

fig.show()
#test on 3 samples

out = net.predict(x_test[:1])



print('true values: ')

print(y_test[0:1])



print('\n')

print('predicted values: ')

out_int = [abs(np.round(x)) for x in out]

print(out_int)