import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
np.random.seed(42)
#Simple Neuron

class Neuron(object):

    def __init__(self, num_inputs, activation):

        super().__init__()

        self.weight = np.random.uniform(size = num_inputs, low = -1.0, high = 1.0)

        self.bias = np.random.uniform(size = 1, low = -1., high = 1.)

        self.activation = activation

        

    def forward(self, x):

        z = np.dot(x, self.weight)+self.bias

        return self.activation(z)
input_size = 3

#ReLU activation function is used here

function = lambda y: 0 if y<=0 else 1



perceptron = Neuron(num_inputs = input_size, activation = function)

print("Perceptron's random weights = {}, and random bias = {}".format(

perceptron.weight, perceptron.bias))
x = np.random.rand(input_size).reshape(1, input_size)

print('Input vector : {}'.format(x))
y = perceptron.forward(x)

print('Output value = {}'.format(y))
class FullyConnected(object):

    def __init__(self, num_inputs, layer_size, activation_function, derivated_activation_function = None):

        super().__init__()

        self.W = np.random.standard_normal((num_inputs, layer_size))

        self.b = np.random.standard_normal(layer_size)

        self.size = layer_size

        self.activation = activation_function

        self.derivated = derivated_activation_function

        self.x, self.y = None, None

        self.dL_dW, self.dL_db = None, None

    

    def forward(self, x):

        z = np.dot(x, self.W) + self.b

        self.y = self.activation(z)

        self.x = x

        return self.y

    

    def backward(self, dL_dy):

        dy_dz = self.derivated(self.y)  

        dL_dz = (dL_dy * dy_dz) 

        dz_dw = self.x.T

        dz_dx = self.W.T

        dz_db = np.ones(dL_dy.shape[0]) 



        self.dL_dW = np.dot(dz_dw, dL_dz)

        self.dL_db = np.dot(dz_db, dL_dz)



        dL_dx = np.dot(dL_dz, dz_dx)

        return dL_dx

    

    def optimize(self, epsilon):

        self.W -= epsilon * self.dL_dW

        self.b -= epsilon * self.dL_db
input_size = 2

neurons = 3

function = lambda y: np.maximum(y, 0)



layer = FullyConnected(num_inputs = input_size, layer_size = neurons, activation_function = function)

x1 = np.random.uniform(-1, 1, 2).reshape(1, 2)

print("Input vector #1: {}".format(x1))
x2 = np.random.uniform(-1, 1, 2).reshape(1, 2)

print("Input vector #2: {}".format(x2))
y1 = layer.forward(x1)

print("Layer's output value given `x1` : {}".format(y1))
y2 = layer.forward(x2)

print("Layer's output value given `x2` : {}".format(y2))
x12 = np.concatenate((x1, x2))  # stack of input vectors, of shape `(2, 2)`

y12 = layer.forward(x12)

print("Layer's output value given `[x1, x2]` :\n{}".format(y12))
def sigmoid(x):

    y = 1/(1 + np.exp(-x))

    return y



def derivated_sigmoid(y):

    return y*(1-y)
def loss_L2(pred, target):            

    return np.sum(np.square(pred - target))



def derivated_loss_L2(pred, target):

    return 2*(pred - target)
def binary_cross_entropy(pred, target):

    return -np.mean(np.multiply(np.log(pred), target) + np.multiply(np.log(1 - pred), (1 - target)))



def derivated_binary_cross_entropy(pred, target):

    return (pred - target) / (pred * (1-pred))
#Simple Neural Network

class SimpleNetwork(object):

    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32),

                 activation_function=sigmoid, derivated_activation_function=derivated_sigmoid,

                 loss_function=loss_L2, derivated_loss_function=derivated_loss_L2):

        super().__init__()

        layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]

        self.layers = [

            FullyConnected(layer_sizes[i], layer_sizes[i + 1], 

                                activation_function, derivated_activation_function)

            for i in range(len(layer_sizes) - 1)]

        self.loss_function = loss_function

        self.derivated_loss_function = derivated_loss_function

    

    def forward(self, x):

        for layer in self.layers: 

            x = layer.forward(x)

        return x

    

    def predict(self, x):

        estimations = self.forward(x)

        best_class = np.argmax(estimations)

        return best_class

    

    def backward(self, dL_dy):

        for layer in reversed(self.layers): 

            dL_dy = layer.backward(dL_dy)

        return dL_dy



    def optimize(self, epsilon):

        for layer in self.layers:            

            layer.optimize(epsilon)

            

    def evaluate_accuracy(self, X_val, y_val):

        num_corrects = 0

        for i in range(len(X_val)):

            pred_class = self.predict(X_val[i])

            if pred_class == y_val[i]:

                num_corrects += 1

        return num_corrects / len(X_val)

    

    def train(self, X_train, y_train, X_val=None, y_val=None, 

              batch_size=32, num_epochs=5, learning_rate=1e-3, print_frequency=20):

        num_batches_per_epoch = len(X_train) // batch_size

        do_validation = X_val is not None and y_val is not None

        losses, accuracies = [], []

        for i in range(num_epochs): 

            epoch_loss = 0

            for b in range(num_batches_per_epoch):  # for each batch composing the dataset

                # Get batch:

                batch_index_begin = b * batch_size

                batch_index_end = batch_index_begin + batch_size

                x = X_train[batch_index_begin: batch_index_end]

                targets = y_train[batch_index_begin: batch_index_end]

                # Optimize on batch:

                predictions = y = self.forward(x)  # forward pass

                L = self.loss_function(predictions, targets)  # loss computation

                dL_dy = self.derivated_loss_function(predictions, targets)  # loss derivation

                self.backward(dL_dy)  # back-propagation pass

                self.optimize(learning_rate)  # optimization of the NN

                epoch_loss += L



            # Logging training loss and validation accuracy, to follow the training:

            epoch_loss /= num_batches_per_epoch

            losses.append(epoch_loss)

            if do_validation:

                accuracy = self.evaluate_accuracy(X_val, y_val)

                accuracies.append(accuracy)

            else:

                accuracy = np.NaN

            if i % print_frequency == 0 or i == (num_epochs - 1):

                print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(

                    i, epoch_loss, accuracy * 100))

        return losses, accuracies
import mnist

X_train, y_train = mnist.train_images(), mnist.train_labels()

X_test,  y_test  = mnist.test_images(), mnist.test_labels()

num_classes = 10
img_idx = np.random.randint(0, X_test.shape[0])

plt.imshow(X_test[img_idx], cmap=matplotlib.cm.binary)

plt.axis("off")

plt.show()
X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)
X_train, X_test = X_train / 255., X_test / 255.

print("Normalized pixel values between {} and {}".format(X_train.min(), X_train.max()))
y_train = np.eye(num_classes)[y_train]
mnist_classifier = SimpleNetwork(num_inputs=X_train.shape[1], 

                                 num_outputs=num_classes, hidden_layers_sizes=[64, 32])
predictions = mnist_classifier.forward(X_train)                         

loss_untrained = mnist_classifier.loss_function(predictions, y_train)   



accuracy_untrained = mnist_classifier.evaluate_accuracy(X_test, y_test) 

print("Untrained : training loss = {:.6f} | val accuracy = {:.2f}%".format(

    loss_untrained, accuracy_untrained * 100))
losses, accuracies = mnist_classifier.train(X_train, y_train, X_test, y_test, 

                                            batch_size=30, num_epochs=500)
losses, accuracies = [loss_untrained] + losses, [accuracy_untrained] + accuracies

fig, ax_loss = plt.subplots()



color = 'red'

ax_loss.set_xlim([0, 510])

ax_loss.set_xlabel('Epochs')

ax_loss.set_ylabel('Training Loss', color=color)

ax_loss.plot(losses, color=color)

ax_loss.tick_params(axis='y', labelcolor=color)



ax_acc = ax_loss.twinx()  # instantiate a second axes that shares the same x-axis

color = 'blue'

ax_acc.set_xlim([0, 510])

ax_acc.set_ylim([0, 1])

ax_acc.set_ylabel('Val Accuracy', color=color)

ax_acc.plot(accuracies, color=color)

ax_acc.tick_params(axis='y', labelcolor=color)



fig.tight_layout()

plt.show()
predicted_class = mnist_classifier.predict(np.expand_dims(X_test[img_idx], 0))

print('Predicted class: {}; Correct class: {}'.format(predicted_class, y_test[img_idx]))