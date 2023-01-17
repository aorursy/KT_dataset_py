import numpy as np



class Sgd():

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):

        updated_weight = weight_tensor - (self.learning_rate) * gradient_tensor

        return updated_weight
class FullyConnected():

    

    def __init__(self, input_size, output_size):

        self.input_size = input_size

        self.output_size = output_size

        weights = np.random.rand(self.input_size + 1, self.output_size)

        self.weights = weights

        self._opt = None

        

    def forward(self, input_tensor):

        batch_size = input_tensor.shape[0]

        bias_row = np.ones((batch_size,1))

        X = np.concatenate((input_tensor, bias_row), axis=1)

        W = self.weights

        output = np.dot(X, W)

        self.stored = X

        self.weights = W

        return output

    

    def getopt(self):

        return self._opt



    def set_optimizer(self, opt):

        self._opt = opt



    optimizer = property(getopt,set_optimizer)



    def backward(self, error_tensor):

        x = self.stored

        errorpre = np.dot(error_tensor,self.weights[0:self.weights.shape[0]-1,:].T)

        self.gradient_tensor = np.dot(x.T, error_tensor)

        if self.optimizer is not None:

            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_tensor)

        return errorpre



    @property

    def gradient_weights(self):

        return self.gradient_tensor
class ReLU():

    def __init__(self):

        pass



    def forward(self, input_tensor):

        relu = lambda input_tensor: input_tensor * (input_tensor >= 0).astype(float)

        out = relu(input_tensor)

        self.store = input_tensor

        return out



    def backward(self, error_tensor):

        x = self.store

        output = error_tensor * (x > 0)

        return output
class SoftMax():

    def __init__(self):

        pass

    def forward(self, input_tensor):

        x = input_tensor

        xdash = np.exp(x - np.max(x))

        xdashsum = np.sum(xdash, axis=1)

        self.store = xdash/xdashsum[:, None]

        return xdash/xdashsum[:, None]

    

    def backward(self, error_tensor):

       e = np.zeros(error_tensor.shape)

       for i in range(error_tensor.shape[0]):

           sum = np.sum(np.multiply(error_tensor[i,:], self.store[i,:]))

           e[i,:]=np.multiply(self.store[i,:],np.subtract(error_tensor[i,:],sum))

       return e
class CrossEntropyLoss():



    def __init__(self):

        pass



    def forward(self, input_tensor, label_tensor):



        loss= np.sum(label_tensor*(-np.log(input_tensor + np.finfo('float').eps)))

        self.input = input_tensor

        return loss



    def backward(self, label_tensor):

        back = -np.divide(label_tensor,self.input)

        return back

import copy

class NeuralNetwork():

    def __init__(self, optimizer):

        self.optimizer = optimizer

        self.loss = []

        self.layers = []

        self.data_layer = None

        self.loss_layer = None



    def forward(self):

        input_tensor, label_tensor = self.data_layer.forward()

        iptensor = input_tensor

        self.label_tensor = label_tensor

        for layer in self.layers:

            iptensor = layer.forward(iptensor)

        self.output = iptensor

        loss = self.loss_layer.forward(iptensor, self.label_tensor)

        return loss



    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):

            error_tensor = layer.backward(error_tensor)



    def append_trainable_layer(self, layer):

        dcopy = copy.deepcopy(self.optimizer)

        layer.optimizer = dcopy

        self.layers.append(layer)





    def train(self, iterations):

        for i in range(iterations):

            self.forward()

            loss = self.loss_layer.forward(self.output, self.label_tensor)

            self.loss.append(loss)

            self.backward()



    def test(self, input_tensor):



        current_input_tensor = input_tensor

        for layer in self.layers:

            current_input_tensor = layer.forward(current_input_tensor)

        return current_input_tensor
from sklearn.datasets import load_iris

from sklearn.preprocessing import OneHotEncoder

from random import shuffle

import matplotlib.pyplot as plt



class IrisData:

    def __init__(self, batch_size):

        self.batch_size = batch_size

        self._data = load_iris()

        self._label_tensor = OneHotEncoder(sparse=False).fit_transform(self._data.target.reshape(-1, 1))

        self._input_tensor = self._data.data

        self._input_tensor /= np.abs(self._input_tensor).max()



        self.split = int(self._input_tensor.shape[0]*(2/3))  # train / test split  == number of samples in train set



        self._input_tensor, self._label_tensor = shuffle_data(self._input_tensor, self._label_tensor)

        self._input_tensor_train = self._input_tensor[:self.split, :]

        self._label_tensor_train = self._label_tensor[:self.split, :]

        self._input_tensor_test = self._input_tensor[self.split:, :]

        self._label_tensor_test = self._label_tensor[self.split:, :]



        self._current_forward_idx_iterator = self._forward_idx_iterator()



    def _forward_idx_iterator(self):

        num_iterations = int(np.ceil(self.split / self.batch_size))

        idx = np.arange(self.split)

        while True:

            this_idx = np.random.choice(idx, self.split, replace=False)

            for i in range(num_iterations):

                yield this_idx[i * self.batch_size:(i + 1) * self.batch_size]



    def forward(self):

        idx = next(self._current_forward_idx_iterator)

        return self._input_tensor_train[idx, :], self._label_tensor_train[idx, :]



    def get_test_set(self):

        return self._input_tensor_test, self._label_tensor_test
net = NeuralNetwork(Sgd(1e-3))

categories = 3

input_size = 4

net.data_layer = IrisData(50)

net.loss_layer = CrossEntropyLoss()



fcl_1 = FullyConnected(input_size, categories)

net.append_trainable_layer(fcl_1)

net.layers.append(ReLU())

fcl_2 = FullyConnected(categories, categories)

net.append_trainable_layer(fcl_2)

net.layers.append(SoftMax())



net.train(4000)

plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')

plt.plot(net.loss, '-x')

plt.show()



data, labels = net.data_layer.get_test_set()



results = net.test(data)

index_maximum = np.argmax(results, axis=1)

one_hot_vector = np.zeros_like(results)

for i in range(one_hot_vector.shape[0]):

    one_hot_vector[i, index_maximum[i]] = 1



    correct = 0.

    wrong = 0.

for column_results, column_labels in zip(one_hot_vector, labels):

    if column_results[column_labels > 0].all() > 0:

        correct += 1

    else:

        wrong += 1



accuracy = correct / (correct + wrong)

print('\nOn the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')