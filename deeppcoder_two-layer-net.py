# A bit of setup

from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import sys

sys.path.append('/kaggle/input/neural-net')

import os

from neural_net import TwoLayerNet

from past.builtins import xrange



%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# for auto-reloading external modules

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2



def rel_error(x, y):

    """ returns relative error """

    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
# Create a small net and some toy data to check your implementations.

# Note that we set the random seed for repeatable experiments.



def __init__(self, input_size, hidden_size, output_size, std=1e-4):

    """

    Initialize the model. Weights are initialized to small random values and

    biases are initialized to zero. Weights and biases are stored in the

    variable self.params, which is a dictionary with the following keys:



    W1: First layer weights; has shape (D, H)

    b1: First layer biases; has shape (H,)

    W2: Second layer weights; has shape (H, C)

    b2: Second layer biases; has shape (C,)



    Inputs:

    - input_size: The dimension D of the input data.

    - hidden_size: The number of neurons H in the hidden layer.

    - output_size: The number of classes C.

    """

    #  randn函数返回一个或一组样本，具有标准正态分布。

    self.params = {}

    self.params['W1'] = std * np.random.randn(input_size, hidden_size)

    self.params['b1'] = np.zeros(hidden_size)

    self.params['W2'] = std * np.random.randn(hidden_size, output_size)

    self.params['b2'] = np.zeros(output_size)





input_size = 4

hidden_size = 10

num_classes = 3

num_inputs = 5



def init_toy_model():

    np.random.seed(0)

    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)



def init_toy_data():

    np.random.seed(1)

    X = 10 * np.random.randn(num_inputs, input_size)

    y = np.array([0, 1, 2, 2, 1])

    return X, y



net = init_toy_model()

X, y = init_toy_data()


scores = net.loss(X)

print('Your scores:')

print(scores)

print()

print('correct scores:')

correct_scores = np.asarray([

  [-0.81233741, -1.27654624, -0.70335995],

  [-0.17129677, -1.18803311, -0.47310444],

  [-0.51590475, -1.01354314, -0.8504215 ],

  [-0.15419291, -0.48629638, -0.52901952],

  [-0.00618733, -0.12435261, -0.15226949]])

print(correct_scores)

print()



# The difference should be very small. We get < 1e-7

print('Difference between your scores and correct scores:')

print(np.sum(np.abs(scores - correct_scores)))
loss, _ = net.loss(X, y, reg=0.05)

correct_loss = 1.30378789133



# should be very small, we get < 1e-12

print('Difference between your loss and correct loss:')

print(np.sum(np.abs(loss - correct_loss)))
from gradient_check import eval_numerical_gradient



# Use numeric gradient checking to check your implementation of the backward pass.

# If your implementation is correct, the difference between the numeric and

# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.



loss, grads = net.loss(X, y, reg=0.05)



# these should all be less than 1e-8 or so

for param_name in grads:

    f = lambda W: net.loss(X, y, reg=0.05)[0]

    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)

    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
net = init_toy_model()

stats = net.train(X, y, X, y,

            learning_rate=1e-1, reg=5e-6,

            num_iters=100, verbose=False)



print('Final training loss: ', stats['loss_history'][-1])



# plot the loss history

plt.plot(stats['loss_history'])

plt.xlabel('iteration')

plt.ylabel('training loss')

plt.title('Training Loss history')

plt.show()
import pandas

def get_mnist_data(num_training=58000, num_validation=2000, num_test=10000):

    """

    Load the Fashion_mnist dataset from disk and perform preprocessing to prepare

    it for the two-layer neural net classifier. These are the same steps as

    we used for the SVM, but condensed to a single function.  

    """

    # Load the raw Fashion_mnis data

    with open("../input/fashionmnist/fashion-mnist_test.csv", "r") as f:

        test_data = pandas.read_csv(f).values

    with open("../input/fashionmnist/fashion-mnist_train.csv", "r") as f:

        train_data = pandas.read_csv(f).values

    

    X_train = train_data[:, 1:].astype(np.float32)

    y_train = train_data[:, 0]



    X_test = train_data[:, 1:].astype(np.float32)

    y_test = train_data[:, 0]



    

    # Subsample the data

    mask = list(range(num_training, num_training + num_validation))

    X_val = X_train[mask]

    y_val = y_train[mask]

    mask = list(range(num_training))

    X_train = X_train[mask]

    y_train = y_train[mask]

    mask = list(range(num_test))

    X_test = X_test[mask]

    y_test = y_test[mask]



    # Normalize the data: subtract the mean image

    mean_image = np.mean(X_train, axis=0).astype('uint8')

    X_train -= mean_image

    X_val -= mean_image

    X_test -= mean_image



    # Reshape data to rows

    X_train = X_train.reshape(num_training, -1)

    X_val = X_val.reshape(num_validation, -1)

    X_test = X_test.reshape(num_test, -1)



    return X_train, y_train, X_val, y_val, X_test, y_test





# Invoke the above function to get our data.

X_train, y_train, X_val, y_val, X_test, y_test = get_mnist_data()

print('Train data shape: ', X_train.shape)

print('Train labels shape: ', y_train.shape)

print('Validation data shape: ', X_val.shape)

print('Validation labels shape: ', y_val.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)
input_size = 28*28

hidden_size = 50

num_classes = 10

net = TwoLayerNet(input_size, hidden_size, num_classes)



# Train the network

stats = net.train(X_train, y_train, X_val, y_val,

            num_iters=1000, batch_size=200,

            learning_rate=1e-4, learning_rate_decay=0.95,

            reg=0.25, verbose=True)



# Predict on the validation set

val_acc = (net.predict(X_val) == y_val).mean()

print('Validation accuracy: ', val_acc)



# Plot the loss function and train / validation accuracies

plt.subplot(2, 1, 1)

plt.plot(stats['loss_history'])

plt.title('Loss history')

plt.xlabel('Iteration')

plt.ylabel('Loss')



plt.subplot(2, 1, 2)

plt.plot(stats['train_acc_history'], label='train')

plt.plot(stats['val_acc_history'], label='val')

plt.title('Classification accuracy history')

plt.xlabel('Epoch')

plt.ylabel('Clasification accuracy')

plt.show()
best_net = None # store the best model into this 



hidden_size = [25, 40, 55, 80, 100]

results = {}

best_val_acc = 0

best_net = None



learning_rates = np.array([0.4, 0.6, 0.8]) * 1e-3

regularization_strengths = [0.10, 0.25, 0.75]

print('tot number =', len(hidden_size) * len(learning_rates) * len(regularization_strengths))

cnt = 0

for hs in hidden_size:

    for lr in learning_rates:

        for reg in regularization_strengths:

            cnt += 1

            print(cnt, end = ':')

            net = TwoLayerNet(input_size, hs, num_classes)

            

            stats = net.train(X_train, y_train, X_val, y_val,

                              num_iters=1500, batch_size=200,

                              learning_rate=lr, learning_rate_decay=0.95,

                              reg= reg, verbose=False)

            val_acc = (net.predict(X_val) == y_val).mean()

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                best_net = net         

            print('hs %d lr %e reg %e val accuracy: %f' % (hs, lr, reg,  val_acc))

            results[(hs,lr,reg)] = val_acc



print ('best validation accuracy achieved during cross-validation: %f' % best_val_acc)

#################################################################################

#                               END OF YOUR CODE                                #

#################################################################################
best_net2 = None # store the best model into this 

hidden_size = [20, 25, 40]

results2 = {}

best_val_acc2 = 0

best_net2 = None



learning_rates = np.array([0.2, 0.4, 0.6]) * 1e-3

regularization_strengths = [0.10, 0.25, 0.5]

number_iteration = [2500, 4000]

print('tot number =', len(hidden_size) * len(learning_rates) * len(regularization_strengths) * len(number_iteration) )

cnt = 0

for hs in hidden_size:

    for lr in learning_rates:

        for reg in regularization_strengths:

            for num_it in number_iteration:

                cnt += 1

                print(cnt, end = ':')

                net = TwoLayerNet(input_size, hs, num_classes)



                stats = net.train(X_train, y_train, X_val, y_val,

                                  num_iters=num_it, batch_size=200,

                                  learning_rate=lr, learning_rate_decay=0.95,

                                  reg= reg, verbose=False)

                val_acc = (net.predict(X_val) == y_val).mean()

                if val_acc > best_val_acc2:

                    best_val_acc2 = val_acc

                    best_net2 = net         

                print('hs %d lr %e reg %e num_it %d val accuracy: %f' % (hs, lr, reg, num_it,  val_acc))

                results2[(hs,lr,reg,num_it)] = val_acc



print ('best validation accuracy achieved during cross-validation: %f' % best_val_acc2)

#################################################################################

#                               END OF YOUR CODE                                #

#################################################################################
# visualize the weights of the best network

from MS326.vis_utils import visualize_grid

def show_net_weights(net):

    W1 = net.params['W1']

    W1 = W1.reshape(28,28,-1).transpose(2,0,1)

#     W1

    plt.imshow(visualize_grid(W1,padding=3).astype('uint8'))

    plt.gca().axis('off')

    plt.show()

# show_net_weights(net)

show_net_weights(best_net2)
test_acc = (best_net2.predict(X_test) == y_test).mean()

print('Test accuracy: ', test_acc)