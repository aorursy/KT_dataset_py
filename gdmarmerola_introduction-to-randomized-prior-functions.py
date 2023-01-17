!pip install tf-nightly-gpu-2.0-preview tfp-nightly

!pip install -q pydot

!apt-get install graphviz
# plotting inline

%matplotlib inline



# importing necessary modules

import keras

import random

import numpy as np

import pandas as pd

import scipy.stats as sp

import matplotlib.pyplot as plt

import tensorflow as tf

from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Activation, concatenate, Input, Embedding

from tensorflow.keras.layers import Reshape, Concatenate, BatchNormalization, Dropout, Add, Lambda

from tensorflow.keras.layers import add

from tensorflow.keras.optimizers import Adam, RMSprop

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from sklearn.ensemble import BaggingRegressor

from copy import deepcopy



# turning off automatic plot showing, and setting style

plt.ioff()

plt.style.use('bmh')
# setting seed 

np.random.seed(42)



# generating big and small datasets

X = np.random.uniform(0.0, 0.5, 100).reshape(-1,1)



# let us generate a grid to check how models fit the data

x_grid = np.linspace(-5, 5, 1000).reshape(-1,1)



# defining the function

noise = sp.norm(0.00, 0.02)

target_toy = lambda x: (x + 0.3*np.sin(2*np.pi*(x + noise.rvs(1)[0])) + 

                        0.3*np.sin(4*np.pi*(x + noise.rvs(1)[0])) + 

                        noise.rvs(1)[0] - 0.5)



# runnning the target

y = np.array([target_toy(e) for e in X])
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.8)

plt.title('Simple 1D example with toy data by Blundell et. al (2015)')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.xlim(-0.5,1.0); plt.ylim(-1.0,1.0)

plt.legend();

plt.show()
# prior network output #



# shared input of the network

net_input = Input(shape=(1,),name='input')



# let us build the prior network with five layers

prior_net = Sequential([Dense(16,'elu',kernel_initializer='glorot_normal',trainable=False),

                        Dense(16,'elu',kernel_initializer='glorot_normal',trainable=False)],

                       name='prior_net')(net_input)



# prior network output

prior_output = Dense(1,'linear',kernel_initializer='glorot_normal',

                     trainable=False, name='prior_out')(prior_net)



# compiling a model for this network

prior_model = Model(inputs=net_input, outputs=prior_output)



# let us score the network and plot the results

prior_preds = 3 * prior_model.predict(x_grid)
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.8)

plt.plot(x_grid, prior_preds, label='prior net (p)')

plt.title('Predictions of the prior network: random function')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.xlim(-0.5,1.0); plt.ylim(-1.0,1.0)

plt.legend();

plt.show()
# adding trainable network #



# trainable network body

trainable_net = Sequential([Dense(16,'elu'),

                            Dense(16,'elu')],

                           name='trainable_net')(net_input)



# trainable network output

trainable_output = Dense(1,'linear',name='trainable_out')(trainable_net)
# using a lambda layer so we can control the weight (beta) of the prior network

prior_scale = Lambda(lambda x: x * 3.0, name='prior_scale')(prior_output)



# lastly, we use a add layer to add both networks together and get Q

add_output = add([trainable_output, prior_scale], name='add')



# defining the model and compiling it

model = Model(inputs=net_input, outputs=add_output)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# checking final architecture

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model).create(prog='dot', format='svg'))
# let us fit the model

model.fit(X, y, epochs=2000, batch_size=100, verbose=0)



# let us get the individual output of the trainable net

trainable_model = Model(inputs=model.input, outputs=model.get_layer('trainable_out').output)
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.8)

plt.plot(x_grid, 3 * prior_model.predict(x_grid), label='prior net (p)')

plt.plot(x_grid, trainable_model.predict(x_grid), label='trainable net (f)')

plt.plot(x_grid, model.predict(x_grid), label='resultant (Q)')

plt.title('Adding the trainable net, and testing our full Keras randomized prior functions model')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.xlim(-0.5,1.0); plt.ylim(-1.0,1.0)

plt.legend();

plt.show()
# function to get a randomized prior functions model

def get_randomized_prior_nn():



    # shared input of the network

    net_input = Input(shape=(1,), name='input')



    # trainable network body

    trainable_net = Sequential([Dense(16,'elu'),

                                Dense(16,'elu')], 

                               name='trainable_net')(net_input)

    

    # trainable network output

    trainable_output = Dense(1, 'linear', name='trainable_out')(trainable_net)



    # prior network body - we use trainable=False to keep the network output random 

    prior_net = Sequential([Dense(16,'elu',kernel_initializer='glorot_normal',trainable=False),

                            Dense(16,'elu',kernel_initializer='glorot_normal',trainable=False)], 

                           name='prior_net')(net_input)

    

    # prior network output

    prior_output = Dense(1, 'linear', kernel_initializer='glorot_normal', trainable=False, name='prior_out')(prior_net)

    

    # using a lambda layer so we can control the weight (beta) of the prior network

    prior_output = Lambda(lambda x: x * 3.0, name='prior_scale')(prior_output)



    # lastly, we use a add layer to add both networks together and get Q

    add_output = add([trainable_output, prior_output], name='add')



    # defining the model and compiling it

    model = Model(inputs=net_input, outputs=add_output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    

    # returning the model 

    return model
# wrapping our base model around a sklearn estimator

base_model = KerasRegressor(build_fn=get_randomized_prior_nn, 

                            epochs=3000, batch_size=100, verbose=0)



# create a bagged ensemble of 10 base models

bag = BaggingRegressor(base_estimator=base_model, n_estimators=9, verbose=2)
# fitting the ensemble

bag.fit(X, y.ravel())
# individual predictions on the grid of values

y_grid = np.array([e.predict(x_grid.reshape(-1,1)) for e in bag.estimators_]).T

trainable_grid = np.array([Model(inputs=e.model.input,outputs=e.model.get_layer('trainable_out').output).predict(x_grid.reshape(-1,1)) for e in bag.estimators_]).T

prior_grid = np.array([Model(inputs=e.model.input,outputs=e.model.get_layer('prior_scale').output).predict(x_grid.reshape(-1,1)) for e in bag.estimators_]).T
# let us check the toy data

plt.figure(figsize=[16,12], dpi=200)



# loop for plotting predictions of each head 

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.plot(X, y, 'kx', label='Toy data', alpha=0.8)

    plt.plot(x_grid, prior_grid[0,:,i], label='prior net (p)')

    plt.plot(x_grid, trainable_grid[0,:,i], label='trainable net (f)')

    plt.plot(x_grid, y_grid[:,i], label='resultant (Q)')

    plt.title('Ensemble: Model #{}'.format(i+1), fontsize=14)

    plt.xlabel('$x$'); plt.ylabel('$y$')

    plt.xlim(-0.5,1.0); plt.ylim(-1.0,1.0)

plt.tight_layout()

plt.show();
# computing mean and stddev

mean = np.array(y_grid).mean(axis=1)

std = np.array(y_grid).std(axis=1)



# opening figure

fig = plt.figure(figsize=[12,5], dpi=150)



# title of the plot

fig.suptitle('Uncertainty estimates given by bootstrapped ensemble of neural networks with randomized priors', verticalalignment='center')



# first subplot, 

plt.subplot(1, 2, 1)



# let us plot the training data

plt.plot(X, y, 'kx',  label='Toy data')

plt.title('Mean and Deviation', fontsize=12)

plt.xlim(-0.5, 1.0); plt.ylim(-1.5, 1.5)

plt.legend()



# plotting predictive mean and deviation

plt.plot(x_grid, mean, 'r--', linewidth=1.5)

plt.fill_between(x_grid.reshape(1,-1)[0], mean - std, mean + std, alpha=0.5, color='red')

plt.fill_between(x_grid.reshape(1,-1)[0], mean + 2*std, mean - 2*std, alpha=0.2, color='red')



# second subplot

plt.subplot(1, 2, 2)



# let us plot the training data

plt.plot(X, y, 'kx', label='Toy data')

plt.title('Samples', fontsize=12)

plt.xlim(-0.5, 1.0); plt.ylim(-1.5, 1.5)

plt.legend()



# loop for plotting predictions of each head 

for i in range(9):

    plt.plot(x_grid, y_grid[:,i], linestyle='--', linewidth=1.5)

    

# showing

plt.show();
# function to get a randomized prior functions model

def get_regular_nn():



    # shared input of the network

    net_input = Input(shape=(1,), name='input')



    # trainable network body

    trainable_net = Sequential([Dense(16, 'elu'),

                                Dense(16, 'elu')], 

                               name='trainable_net')(net_input)

    

    # trainable network output

    trainable_output = Dense(1, activation='linear', name='trainable_out')(trainable_net)



    # defining the model and compiling it

    model = Model(inputs=net_input, outputs=trainable_output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    

    # returning the model 

    return model
# wrapping our base models around a sklearn estimator

base_rpf = KerasRegressor(build_fn=get_randomized_prior_nn, 

                          epochs=3000, batch_size=100, verbose=0)

base_reg = KerasRegressor(build_fn=get_regular_nn, 

                          epochs=3000, batch_size=100, verbose=0)



# our models

prior_but_no_boostrapping = BaggingRegressor(base_rpf, n_estimators=9, bootstrap=False)

bootstrapping_but_no_prior = BaggingRegressor(base_reg, n_estimators=9)

no_prior_and_no_boostrapping = BaggingRegressor(base_reg, n_estimators=9, bootstrap=False)



# fitting the models

prior_but_no_boostrapping.fit(X, y.ravel())

bootstrapping_but_no_prior.fit(X, y.ravel())

no_prior_and_no_boostrapping.fit(X, y.ravel())
# individual predictions on the grid of values

y_grid = np.array([e.predict(x_grid.reshape(-1,1)) for e in bag.estimators_]).T

y_grid_1 = np.array([e.predict(x_grid.reshape(-1,1)) for e in prior_but_no_boostrapping.estimators_]).T

y_grid_2 = np.array([e.predict(x_grid.reshape(-1,1)) for e in bootstrapping_but_no_prior.estimators_]).T

y_grid_3 = np.array([e.predict(x_grid.reshape(-1,1)) for e in no_prior_and_no_boostrapping.estimators_]).T
# opening figure

fig = plt.figure(figsize=[12,9], dpi=150)



# title of the plot

fig.suptitle('Bootstrapping and priors: impact of model components on result', verticalalignment='center')



# second subplot

plt.subplot(2, 2, 1)



# let us plot the training data

plt.plot(X, y, 'kx', label='Toy data')

plt.title('Full model with priors and bootstrap', fontsize=12)

plt.xlim(-0.5, 1.0); plt.ylim(-1.5, 1.5)

plt.legend()



# loop for plotting predictions of each head 

for i in range(9):

    plt.plot(x_grid, y_grid[:,i], linestyle='--', linewidth=1.5)



# second subplot

plt.subplot(2, 2, 2)



# let us plot the training data

plt.plot(X, y, 'kx', label='Toy data')

plt.title('No bootrapping, but use of priors', fontsize=12)

plt.xlim(-0.5, 1.0); plt.ylim(-1.5, 1.5)

plt.legend()



# loop for plotting predictions of each head 

for i in range(9):

    plt.plot(x_grid, y_grid_1[:,i], linestyle='--', linewidth=1.5)



# second subplot

plt.subplot(2, 2, 3)



# let us plot the training data

plt.plot(X, y, 'kx', label='Toy data')

plt.title('No priors, but use of bootstrapping', fontsize=12)

plt.xlim(-0.5, 1.0); plt.ylim(-1.5, 1.5)

plt.legend()



# loop for plotting predictions of each head 

for i in range(9):

    plt.plot(x_grid, y_grid_2[:,i], linestyle='--', linewidth=1.5)



# second subplot

plt.subplot(2, 2, 4)



# let us plot the training data

plt.plot(X, y, 'kx', label='Toy data')

plt.title('Both bootstrapping and priors turned off', fontsize=12)

plt.xlim(-0.5, 1.0); plt.ylim(-1.5, 1.5)

plt.legend()



# loop for plotting predictions of each head 

for i in range(9):

    plt.plot(x_grid, y_grid_3[:,i], linestyle='--', linewidth=1.5)

    

# showing

plt.show();