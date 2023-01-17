!pip install tf-nightly-gpu-2.0-preview tfp-nightly

!pip install -q pydot

!apt-get install graphviz

!pip install keras-tqdm
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

from keras_tqdm import TQDMNotebookCallback



# turning off automatic plot showing, and setting style

plt.ioff()

plt.style.use('bmh')
# setting seed 

np.random.seed(10)



# generating big and small datasets

X = np.clip(np.random.normal(0.0, 1.0, 1000).reshape(-1,1), -3, 3)



# let us generate a grid to check how models fit the data

x_grid = np.linspace(-5, 5, 1000).reshape(-1,1)



# defining the function - noisy

noise = lambda x: sp.norm(0.00, 0.01 + (x**2)/10)

target_toy = lambda x: (x + 0.3*np.sin(2*np.pi*(x + noise(x).rvs(1)[0])) + 

                        0.3*np.sin(4*np.pi*(x + noise(x).rvs(1)[0])) + 

                        noise(x).rvs(1)[0] - 0.5)



# defining the function - no noise

target_toy_noiseless = lambda x: (x + 0.3*np.sin(2*np.pi*(x)) + 0.3*np.sin(4*np.pi*(x)) - 0.5)



# runnning the target

y = np.array([target_toy(e) for e in X])

y_noiseless = np.array([target_toy_noiseless(e) for e in x_grid])
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)

#plt.plot(x_grid, y_noiseless, 'r--')

plt.title('Data for estimating uncertainty and risk')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.legend();

plt.show()
# function to get a randomized prior functions model

def get_regular_nn():



    # shared input of the network

    net_input = Input(shape=(1,), name='input')



    # trainable network body

    trainable_net = Sequential([Dense(16, 'elu'),

                                Dense(16, 'elu'),

                                Dense(16, 'elu')], 

                               name='layers')(net_input)

    

    # trainable network output

    trainable_output = Dense(1, activation='linear', name='output')(trainable_net)



    # defining the model and compiling it

    model = Model(inputs=net_input, outputs=trainable_output)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])

    

    # returning the model 

    return model
# generating the model

regular_nn = get_regular_nn();



# fitting the model

regular_nn.fit(X, y, batch_size=16, epochs=500, verbose=0)
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)

plt.plot(x_grid, regular_nn.predict(x_grid), label='neural net fit', color='tomato', alpha=0.8)

plt.title('Neural network fit for median expected value')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.xlim(-3.5,3.5); plt.ylim(-5, 3)

plt.legend();

plt.show()
# implementing the tilted (quantile) loss

import tensorflow.keras.backend as K

def tilted_loss(q,y,f):

    e = (y-f)

    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)



# losses for 10th, 50th and 90th percentile

loss_10th_p = lambda y,f: tilted_loss(0.10,y,f)

loss_50th_p = lambda y,f: tilted_loss(0.50,y,f)

loss_90th_p = lambda y,f: tilted_loss(0.90,y,f)
# function to get a randomized prior functions model

def get_quantile_reg_nn():



    # shared input of the network

    net_input = Input(shape=(1,), name='input')



    # trainable network body

    trainable_net = Sequential([Dense(16, 'elu'),

                                Dense(16, 'elu'),

                                Dense(16, 'elu')], 

                               name='shared')(net_input)

    

    # trainable network output

    output_10th_p = Sequential([Dense(8, activation='elu'), 

                                Dense(1, activation='linear')],

                               name='output_10th_p')(trainable_net)

    output_50th_p = Sequential([Dense(8, activation='elu'), 

                                Dense(1, activation='linear')],

                               name='output_50th_p')(trainable_net)

    output_90th_p = Sequential([Dense(8, activation='elu'), 

                                Dense(1, activation='linear')],

                               name='output_90th_p')(trainable_net)

    

    # defining the model and compiling it

    model = Model(inputs=net_input, outputs=[output_10th_p, output_50th_p, output_90th_p])

    model.compile(loss=[loss_10th_p, loss_50th_p, loss_90th_p], optimizer='adam')

    

    # returning the model 

    return model
# checking final architecture

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(get_quantile_reg_nn()).create(prog='dot', format='svg'))
# generating the model

quantile_nn = get_quantile_reg_nn();



# fitting the model

quantile_nn.fit(X, [y]*3, batch_size=16, epochs=500, verbose=0)
# output of the neural net

quantile_output = np.array(quantile_nn.predict(x_grid)).reshape(3, 1000)



# getting quantiles

output_10th_p = quantile_output[0,:]

output_50th_p = quantile_output[1,:]

output_90th_p = quantile_output[2,:]
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)

plt.plot(x_grid, output_10th_p, label='10th and 90th percentile', color='dodgerblue', linewidth=1.8, alpha=0.8)

plt.plot(x_grid, output_50th_p, label='50th percentile', color='tomato', linewidth=1.8, alpha=0.8)

plt.plot(x_grid, output_90th_p, color='dodgerblue', linewidth=1.8, alpha=0.8)

plt.title('Estimating risk: Neural network fit for median, 10th and 90th percentile')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.xlim(-3.5,3.5); plt.ylim(-5, 3)

plt.legend();

plt.show()
# function to get a randomized prior functions model

def get_quantile_reg_rpf_nn():



    # shared input of the network

    net_input = Input(shape=(1,), name='input')



    # trainable network body

    trainable_net = Sequential([Dense(16, 'elu'),

                                Dense(16, 'elu'),

                                Dense(16, 'elu')], 

                               name='trainable_shared')(net_input)

    

    # trainable network outputs

    train_out_1 = Sequential([Dense(8, activation='elu'), 

                              Dense(1, activation='linear')],

                              name='train_out_1')(trainable_net)

    train_out_2 = Sequential([Dense(8, activation='elu'), 

                              Dense(1, activation='linear')],

                              name='train_out_2')(trainable_net)

    train_out_3 = Sequential([Dense(8, activation='elu'), 

                              Dense(1, activation='linear')],

                              name='train_out_3')(trainable_net)

    

    # prior network body

    prior_net = Sequential([Dense(16, 'elu', kernel_initializer='glorot_normal', 

                                  trainable=False),

                            Dense(16, 'elu', kernel_initializer='glorot_normal', 

                                  trainable=False),

                            Dense(16, 'elu', kernel_initializer='glorot_normal', 

                                  trainable=False)], 

                           name='prior_shared')(net_input)



    # prior network outputs

    prior_out_1 = Dense(1, 'elu', kernel_initializer='glorot_normal', 

                        trainable=False, name='prior_out_1')(prior_net)

    prior_out_2 = Dense(1, 'elu', kernel_initializer='glorot_normal', 

                        trainable=False, name='prior_out_2')(prior_net)

    prior_out_3 = Dense(1, 'elu', kernel_initializer='glorot_normal', 

                        trainable=False, name='prior_out_3')(prior_net)



    # using a lambda layer so we can control the weight (beta) of the prior network

    prior_out_1 = Lambda(lambda x: x * 3.0, name='prior_scale_1')(prior_out_1)

    prior_out_2 = Lambda(lambda x: x * 3.0, name='prior_scale_2')(prior_out_2)

    prior_out_3 = Lambda(lambda x: x * 3.0, name='prior_scale_3')(prior_out_3)

    

    # adding all the outputs together

    add_out_1 = add([train_out_1, prior_out_1], name='add_out_1')

    add_out_2 = add([train_out_2, prior_out_2], name='add_out_2')

    add_out_3 = add([train_out_3, prior_out_3], name='add_out_3')

    

    # defining the model and compiling it

    model = Model(inputs=net_input, outputs=[add_out_1, add_out_2, add_out_3])

    model.compile(loss=[loss_10th_p, loss_50th_p, loss_90th_p], optimizer='adam')

    

    # returning the model 

    return model
# checking final architecture

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(get_quantile_reg_rpf_nn()).create(prog='dot', format='svg'))
class MyMultiOutputKerasRegressor(KerasRegressor):

    

    # initializing

    def __init__(self, **kwargs):

        KerasRegressor.__init__(self, **kwargs)

        

    # simpler fit method

    def fit(self, X, y, **kwargs):

        KerasRegressor.fit(self, X, [y]*3, **kwargs)
# wrapping our base model around a sklearn estimator

base_model = MyMultiOutputKerasRegressor(build_fn=get_quantile_reg_rpf_nn, 

                                         epochs=500, batch_size=16, verbose=0)



# create a bagged ensemble of 10 base models

bag = BaggingRegressor(base_estimator=base_model, n_estimators=10, verbose=2)
# fitting the ensemble

bag.fit(X, y)
# output of the neural net

quantile_output = np.array([np.array(e.predict(x_grid)).reshape(3, 1000) for e in bag.estimators_])
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)

plt.plot(x_grid, quantile_output[0,0,:], label='10th and 90th percentile', color='dodgerblue', linewidth=1, alpha=0.5)

for i in range(1,10):

    plt.plot(x_grid, quantile_output[i,0,:], color='dodgerblue', linewidth=1, alpha=0.5)

plt.plot(x_grid, quantile_output[0,1,:], label='50th percentile', color='tomato', linewidth=1, alpha=0.8)

for i in range(1,10):

    plt.plot(x_grid, quantile_output[i,1,:], color='tomato', linewidth=1, alpha=0.8)

for i in range(10):

    plt.plot(x_grid, quantile_output[i,2,:], color='dodgerblue', linewidth=1, alpha=0.5)

plt.title('Estimating risk and uncertainty: Randomized Prior Functions fit for median, 10th and 90th percentile\nShowing individual ensemble members (posterior samples)')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.xlim(-3.5,3.5); plt.ylim(-5, 3)

plt.legend();

plt.show()
# let us take median and quantiles for each parameter #



# median

p50_median = np.quantile(quantile_output[:,1,:], 0.5, axis=0)

p50_90 = np.quantile(quantile_output[:,1,:], 0.9, axis=0)

p50_10 = np.quantile(quantile_output[:,1,:], 0.1, axis=0)



# 10th percentile

p10_median = np.quantile(quantile_output[:,0,:], 0.5, axis=0)

p10_90 = np.quantile(quantile_output[:,0,:], 0.9, axis=0)

p10_10 = np.quantile(quantile_output[:,0,:], 0.1, axis=0)



# 90th percentile

p90_median = np.quantile(quantile_output[:,2,:], 0.5, axis=0)

p90_90 = np.quantile(quantile_output[:,2,:], 0.9, axis=0)

p90_10 = np.quantile(quantile_output[:,2,:], 0.1, axis=0)
# let us check the toy data

plt.figure(figsize=[12,6], dpi=200)



# first plot

plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)

plt.plot(x_grid.reshape(1,-1)[0], p10_median, label='10th and 90th percentile', color='dodgerblue', linewidth=1.5, alpha=0.8)

plt.fill_between(x_grid.reshape(1,-1)[0], p10_10, p10_90, color='dodgerblue', alpha=0.3, label='uncertainty over 10th and 90th percentiles')

plt.plot(x_grid.reshape(1,-1)[0], p90_median, color='dodgerblue', linewidth=1.5, alpha=0.8)

plt.fill_between(x_grid.reshape(1,-1)[0], p90_10, p90_90, color='dodgerblue', alpha=0.3)

plt.plot(x_grid.reshape(1,-1)[0], p50_median, label='50th percentile', color='tomato', linewidth=1.5, alpha=0.8)

plt.fill_between(x_grid.reshape(1,-1)[0], p50_10, p50_90, color='tomato', alpha=0.3, label='uncertainty over median')

plt.title('Estimating risk and uncertainty: Randomized Prior Functions fit for median, 10th and 90th percentile\nShowing median, 10th and 90th percentile across ensemble members')

plt.xlabel('$x$'); plt.ylabel('$y$')

plt.xlim(-3.5,3.5); plt.ylim(-5, 3)

plt.legend();

plt.show()