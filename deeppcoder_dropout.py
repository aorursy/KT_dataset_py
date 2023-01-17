# As usual, a bit of setup

from __future__ import print_function

import time

import numpy as np

import matplotlib.pyplot as plt

import sys

sys.path.append('/kaggle/input/dropoutdata')

import os

from fc_net import *

from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

from solver import Solver

from builtins import range

from past.builtins import xrange



import numpy as np

from random import randrange









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
import pandas

def get_mnist_data(num_training=58000, num_validation=2000, num_test=10000,

                     subtract_mean=True):

    """

    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare

    it for classifiers. These are the same steps as we used for the SVM, but

    condensed to a single function.

    """

    """

    Load the Fashion_mnist dataset from disk and perform preprocessing to prepare

    it for the linear classifier. These are the same steps as we used for the

    SVM, but condensed to a single function.  

    """

    # Load the raw Fashion_mnist data

    with open("../input/fashionmnist/fashion-mnist_test.csv", "r") as f:

        test_data = pandas.read_csv(f).values

    with open("../input/fashionmnist/fashion-mnist_train.csv", "r") as f:

        train_data = pandas.read_csv(f).values



    X_train = train_data[:, 1:].astype(np.float32)

    y_train = train_data[:, 0]



    X_test = train_data[:, 1:].astype(np.float32)

    y_test = train_data[:, 0]

    X_train = np.array(X_train,dtype=np.float)

    X_test = np.array(X_test,dtype=np.float)



    # subsample the data

    mask = list(range(num_training, num_training + num_validation))

    X_val = X_train[mask]

    y_val = y_train[mask]

    mask = list(range(num_training))

    X_train = X_train[mask]

    y_train = y_train[mask]

    mask = list(range(num_test))

    X_test = X_test[mask]

    y_test = y_test[mask]

    

    # Preprocessing: reshape the image data into rows

    X_train = np.reshape(X_train, (X_train.shape[0], -1))

    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    

    # Normalize the data: subtract the mean image

    mean_image = np.mean(X_train, axis = 0).astype('uint8')

    X_train -= mean_image

    X_val -= mean_image

    X_test -= mean_image

    

    # add bias dimension and transform into columns

    X_train =  X_train.reshape(58000,1,28,28)

    X_val = X_val.reshape(-1,1,28,28)

    X_test = X_test.reshape(-1,1,28,28)

    

    # Package data into a dictionary

    return {

      'X_train': X_train, 'y_train': y_train,

      'X_val': X_val, 'y_val': y_val,

      'X_test': X_test, 'y_test': y_test,

    }

data = get_mnist_data()

for k, v in data.items():

  print('%s: ' % k, v.shape)


np.random.seed(326)

x = np.random.randn(500, 500) + 10



for p in [0.25, 0.4, 0.7]:

  out, _ = dropout_forward(x, {'mode': 'train', 'p': p})

  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})



  print('Running tests with p = ', p)

  print('Mean of input: ', x.mean())

  print('Mean of train-time output: ', out.mean())

  print('Mean of test-time output: ', out_test.mean())

  print('Fraction of train-time output set to zero: ', (out == 0).mean())

  print('Fraction of test-time output set to zero: ', (out_test == 0).mean())

  print()
np.random.seed(326)

x = np.random.randn(10, 10) + 10

dout = np.random.randn(*x.shape)



dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}

out, cache = dropout_forward(x, dropout_param)

dx = dropout_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)



# Error should be around e-10 or less

print('dx relative error: ', rel_error(dx, dx_num))
np.random.seed(326)

N, D, H1, H2, C = 2, 15, 20, 30, 10

X = np.random.randn(N, D)

y = np.random.randint(C, size=(N,))



for dropout in [1, 0.75, 0.5]:

  print('Running check with dropout = ', dropout)

  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,

                            weight_scale=5e-2, dtype=np.float64,

                            dropout=dropout, seed=123)



  loss, grads = model.loss(X, y)

  print('Initial loss: ', loss)

  

  # Relative errors should be around e-6 or less; Note that it's fine

  # if for dropout=1 you have W2 error be on the order of e-5.

  for name in sorted(grads):

    f = lambda _: model.loss(X, y)[0]

    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)

    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

  print()
# Train two identical nets, one with dropout and one without

np.random.seed(326)

num_train = 500

small_data = {

  'X_train': data['X_train'][:num_train],

  'y_train': data['y_train'][:num_train],

  'X_val': data['X_val'],

  'y_val': data['y_val'],

}



solvers = {}

dropout_choices = [1, 0.25]

for dropout in dropout_choices:

  model = FullyConnectedNet([500], dropout=dropout)

  print(dropout)



  solver = Solver(model, small_data,

                  num_epochs=25, batch_size=100,

                  update_rule='adam',

                  optim_config={

                    'learning_rate': 5e-4,

                  },

                  verbose=True, print_every=100)

  solver.train()

  solvers[dropout] = solver

  print()
# Plot train and validation accuracies of the two models



train_accs = []

val_accs = []

for dropout in dropout_choices:

  solver = solvers[dropout]

  train_accs.append(solver.train_acc_history[-1])

  val_accs.append(solver.val_acc_history[-1])



plt.subplot(3, 1, 1)

for dropout in dropout_choices:

  plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)

plt.title('Train accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(ncol=2, loc='lower right')

  

plt.subplot(3, 1, 2)

for dropout in dropout_choices:

  plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)

plt.title('Val accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(ncol=2, loc='lower right')



plt.gcf().set_size_inches(15, 15)

plt.show()