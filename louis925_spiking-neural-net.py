# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
class Brain():
    def __init__(self, n_neurons=100, n_inputs=10, n_outputs=10, n_connections=10,
                 low=-1, resting=0, threshold=1, peak=5, decay=0.5, seed=2020):
        """ Create a brain with hidden, input, and output neurons
            Args: 
            - n_neurons [int]: total number of neurons
            - n_inputs [int]: number of neurons having input
            - n_outputs [int]: number of neurons having output
            - n_connections [int]: number of connections per neuron
            - low: minimum potential of neuron
            - resting: resting potential of neuron
            - threshold: threshold potential of neuron
            - peak: peak fire potential (action potential) of neuron
            - decay [float]: decay rate of neuron potential
            - seed [int]: random seed for numpy
            Return: Brain object
        """
        assert n_neurons >= n_outputs + n_inputs
        assert n_neurons >= n_connections
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_connections = n_connections
        self.low = low
        self.resting = resting     
        self.threshold = threshold
        self.peak = peak
        self.decay = decay
        np.random.seed(seed)
        self.mask = self.generate_mask()
        self.initialize_weights()
        self.initialize_neurons()        

    def generate_mask(self):
        """ Determine how the neurons are connected. It will try to generate on average n_connections per neurons.
            No self-connection are allowed.
        """
        threshold = (self.n_connections + 1) / self.n_neurons
        return (np.random.rand(self.n_neurons, self.n_neurons) < threshold) & ~np.identity(self.n_neurons, dtype='bool')
    
    def initialize_weights(self, seed=None):
        """ Initialize the weights of conected neurons to be between -1 and 1. """
        if seed:
            np.random.seed(seed)
        self.weights = (2 * np.random.rand(self.n_neurons, self.n_neurons).astype('float32') - 1) * self.mask
    
    def initialize_neurons(self):
        self.neurons = np.ones(self.n_neurons).astype('float32') * self.resting  # potentials of the neurons
        self.neurons_fire = np.zeros(self.n_neurons, dtype='bool')               # whether the neurons fires or not
    
    def forward_pass(self):
        # compute input potential
        neurons_next = np.dot(self.weights, self.neurons_fire * self.peak) + self.decay * self.neurons
        # If the neuron has just fired, then suppress its potential to the lowest potential
        neurons_next = np.where(self.neurons_fire, self.low, neurons_next)
        neurons_next = np.clip(neurons_next, a_min=self.low, a_max=None)
        # If input potential (x) is greater than threshold, then fire at peak potential, otherwise no change.
        self.neurons = neurons_next
        self.neurons_fire = self.neurons >= self.threshold
    
    def train_hebbian_one_step(self, learning_rate=0.05):
        """ Train with Hebbian like method (unsupervised learning)
            dW_ij = lr * (next_fire_i * M_ij * prev_fire_j - prev_fire_i * M_ij * next_fire_j)
                  = lr * M_ij * (next_fire_i * prev_fire_j - prev_fire_i * next_fire_j)
        """
        prev_fire = self.neurons_fire
        self.forward_pass()
        next_fire = self.neurons_fire
        # dW = learning_rate * next_fire.reshape(-1, 1) * self.weights * next_fire
        dW = learning_rate * self.mask * (
            next_fire.reshape(-1, 1) * prev_fire - 1 * prev_fire.reshape(-1, 1) * next_fire
        )
        self.weights = np.clip(self.weights + dW, a_min=-1, a_max=1)
    
    def set_inputs(self, inputs):
        self.neurons[:self.n_inputs] = inputs / self.decay
    
    def get_outputs(self):
        return self.neurons[self.n_inputs : self.n_inputs + self.n_outputs]
    
    def predict(self, inputs, steps=10, constant_input=False):
        self.set_inputs(inputs)
        for step in range(steps):
            if constant_input: 
                self.set_inputs(inputs)
            self.forward_pass()
        return self.get_outputs()
    
    def predict_history(self, inputs, steps=10, constant_input=False):
        self.set_inputs(inputs)
        hist = np.zeros((steps, self.n_outputs), dtype='float32')
        for step in range(steps):
            if constant_input: 
                self.set_inputs(inputs)
            self.forward_pass()
            hist[step] = self.get_outputs()
        return hist

    def predict_brain_history(self, inputs, steps=10, constant_input=False):
        self.set_inputs(inputs)
        hist = np.zeros((steps, self.n_neurons), dtype='float32')
        for step in range(steps):
            if constant_input: 
                self.set_inputs(inputs)
            self.forward_pass()
            hist[step] = self.neurons
        return hist
    
    def train_hebbian_brain_history(self, inputs, steps=10, lr=0.05, constant_input=False):
        self.set_inputs(inputs)
        hist = np.zeros((steps, self.n_neurons), dtype='float32')
        for step in range(steps):
            if constant_input: 
                self.set_inputs(inputs)
            self.train_hebbian_one_step(lr)
            hist[step] = self.neurons
        return hist
    
    def train_hebbian_brain_batch(self, X, total_steps=1000, steps_per_sample=10, lr=0.05, constant_input=True):
        """ Train Hebbian brain with an array of data X """
        n_samples_train = np.ceil(total_steps / steps_per_sample).astype('int')
        total_steps = n_samples_train * steps_per_sample
        if n_samples_train > len(X):
            X = np.vstack([X] * np.ceil(n_samples_train / len(X)).astype('int'))
        X = X[:n_samples_train]
        print('training steps:', n_samples_train, 'x', steps_per_sample)
        for x in tqdm(X):
            self.set_inputs(x)
            for _ in range(steps_per_sample):
                if constant_input: 
                    self.set_inputs(x)
                self.train_hebbian_one_step(lr)
# Create our brain
brain = Brain(n_neurons=100, n_inputs=10, n_outputs=10, n_connections=10)
brain.initialize_neurons()
display(brain.predict(np.array([5,]+[0]*9), 9, constant_input=False))

brain.initialize_neurons()
display(brain.predict(np.array([0]*9+[1]), 9, constant_input=False))

brain.initialize_neurons()
display(brain.predict(np.array([0]*9+[1]), 9, constant_input=True))
plt.figure(figsize=(10, 4))
plt.subplot(121).set_title('weights')
plt.imshow(brain.weights)
plt.subplot(122).set_title('mask')
plt.imshow(brain.mask)
plt.show()
# History of output neuron fire
brain.initialize_neurons()
plt.imshow(brain.predict_history(np.array([1,]+[0]*9), 30, constant_input=False)); plt.show()
# History of output neuron fire with constant input
brain.initialize_neurons()
plt.imshow(brain.predict_history(np.array([1,]+[0]*9), 30, constant_input=True)); plt.show()
# History of all neurons
brain.initialize_neurons()
plt.imshow(brain.predict_brain_history(np.array([1,]+[0]*9), 90, constant_input=False)); plt.show()
# History of all neurons with constant input
brain.initialize_neurons()
plt.imshow(brain.predict_brain_history(np.array([1,]+[0]*9), 90, constant_input=True)); plt.show()
# non-constant inputs vs outputs at various timesteps
plt.figure(figsize=(12, 6))
steps = [9, 19, 39]
plt_subplot = 101 + 10 * len(steps)
for step in steps:
    pred = np.zeros((10, brain.n_outputs), dtype='float32')
    for i in range(10):
        brain.initialize_neurons()
        pred[i] = brain.predict(np.array([0]*i+[1]+[0]*(9 - i)), step, constant_input=False)
    display(pd.DataFrame(pred))
    plt.subplot(plt_subplot).set_title(f'steps = {step}')
    plt.imshow(pred)
    plt.ylabel('input neuron location'); plt.xlabel('output neuron fire location'); 
    plt_subplot += 1
plt.show()
# constant inputs vs outputs at various timesteps
plt.figure(figsize=(12, 6))
steps = [9, 19, 39]
plt_subplot = 101 + 10 * len(steps)
for step in steps:
    pred = np.zeros((10, brain.n_outputs), dtype='float32')
    for i in range(10):
        brain.initialize_neurons()
        pred[i] = brain.predict(np.array([0]*i+[1]+[0]*(9 - i)), step, constant_input=True)
    display(pd.DataFrame(pred))
    plt.subplot(plt_subplot).set_title(f'steps = {step}')
    plt.imshow(pred)
    plt.ylabel('input neuron location'); plt.xlabel('output neuron fire location'); 
    plt_subplot += 1
plt.show()
# constant inputs vs outputs at various timesteps
plt.figure(figsize=(16, 8))
plt_subplot = [2, 5, 1]
for i in range(10):
    brain.initialize_neurons()
    pred = brain.predict_history(np.array([0]*i+[1]+[0]*(9 - i)), steps=10, constant_input=True)
    #print(pred)
    plt.subplot(*plt_subplot).set_title(f'input = {i}')
    plt.imshow(pred)
    plt.ylabel('timesteps'); plt.xlabel('output neuron fire location'); 
    plt_subplot[2] += 1
plt.show()
# non-constant inputs vs outputs at various timesteps
plt.figure(figsize=(16, 8))
plt_subplot = [2, 5, 1]
for i in range(10):
    brain.initialize_neurons()
    pred = brain.predict_history(np.array([0]*i+[1]+[0]*(9 - i)), steps=10, constant_input=False)
    #print(pred)
    plt.subplot(*plt_subplot).set_title(f'input = {i}')
    plt.imshow(pred)
    plt.ylabel('timesteps'); plt.xlabel('output neuron fire location'); 
    plt_subplot[2] += 1
plt.show()
print('Non-constant input')
plt.figure(figsize=(10, 5))
# with training
brain.initialize_neurons()
brain.initialize_weights(2030)
plt.subplot(1, 2, 1).set_title('with training')
plt.imshow(brain.train_hebbian_brain_history(np.array([1,]+[0]*9), 90, lr=0.05, constant_input=False))

# without training
brain.initialize_neurons()
brain.initialize_weights(2030)
plt.subplot(1, 2, 2).set_title('without training')
plt.imshow(brain.predict_brain_history(np.array([1,]+[0]*9), 90, constant_input=False))
plt.show()
print('Constant input')
plt.figure(figsize=(10, 5))
# with training
brain.initialize_neurons()
brain.initialize_weights(2030)
plt.subplot(1, 2, 1).set_title('with training')
plt.imshow(brain.train_hebbian_brain_history(np.array([1,]+[0]*9), 90, lr=0.05, constant_input=True))

# without training
brain.initialize_neurons()
brain.initialize_weights(2030)
plt.subplot(1, 2, 2).set_title('without training')
plt.imshow(brain.predict_brain_history(np.array([1,]+[0]*9), 90, constant_input=True))
plt.show()
# Weights
plt_bins = np.arange(-1, 1.02, 0.02)
plt.figure(figsize=(12, 4))
plt.title('Weight distributions after training with constant input')
brain.initialize_neurons()
brain.initialize_weights(2030)
plt.hist(brain.weights.reshape(-1)[brain.mask.reshape(-1)], bins=plt_bins, label='before training', alpha=0.7)
brain.train_hebbian_brain_history(np.array([1,]+[0]*9), 90, lr=0.05, constant_input=True)
plt.hist(brain.weights.reshape(-1)[brain.mask.reshape(-1)], bins=plt_bins, label='after training', alpha=0.7)
plt.xlabel('weight'); plt.ylabel('N weights')
plt.legend(); plt.grid(); plt.show()
# Weights
plt_bins = np.arange(-1, 1.02, 0.02)
plt.figure(figsize=(12, 4))
plt.title('Weight distributions after training with non-constant input')
brain.initialize_neurons()
brain.initialize_weights(2030)
plt.hist(brain.weights.reshape(-1)[brain.mask.reshape(-1)], bins=plt_bins, label='before training', alpha=0.7)
brain.train_hebbian_brain_history(np.array([1,]+[0]*9), 90, lr=0.05, constant_input=False)
plt.hist(brain.weights.reshape(-1)[brain.mask.reshape(-1)], bins=plt_bins, label='after training', alpha=0.7)
plt.xlabel('weight'); plt.ylabel('N weights')
plt.legend(); plt.grid(); plt.show()
# Weights change
# plt.figure(figsize=(12, 4))
plt.title('Weight change after training with constant input')
brain.initialize_neurons()
brain.initialize_weights(2030)
W_before = brain.weights
brain.train_hebbian_brain_history(np.array([1,]+[0]*9), 90, lr=0.05, constant_input=True)
dW = brain.weights - W_before
plt.imshow(dW)
plt.xlabel('input'); plt.ylabel('output')
plt.show()
# Weights change
# plt.figure(figsize=(12, 4))
plt.title('Weight change after training with non-constant input')
brain.initialize_neurons()
brain.initialize_weights(2030)
W_before = brain.weights
brain.train_hebbian_brain_history(np.array([1,]+[0]*9), 90, lr=0.05, constant_input=False)
dW = brain.weights - W_before
plt.imshow(dW)
plt.xlabel('input'); plt.ylabel('output')
plt.show()
# weight change for various constant inputs
plt.figure(figsize=(16, 8))
plt_subplot = [2, 5, 1]
for i in range(10):
    brain.initialize_neurons()
    brain.initialize_weights(2030)
    W_before = brain.weights
    brain.train_hebbian_brain_history(np.array([1,]+[0]*9), 90, lr=0.05, constant_input=False)
    dW = brain.weights - W_before
    plt.subplot(*plt_subplot).set_title(f'input = {i}')
    plt.imshow(dW)
    plt.ylabel('output'); plt.xlabel('input'); 
    plt_subplot[2] += 1
plt.show()
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

cols_pixels = [c for c in df_train.columns if c != 'label']

y_train = df_train['label']
x_train = df_train[cols_pixels].values
x_test = df_test[cols_pixels].values

x_train = x_train / 255
x_test = x_test / 255

print(x_train.shape, x_test.shape, y_train.shape)
print(x_train.min(), x_train.max())
# Create our brain
brain_mnist = Brain(n_neurons=2000, n_inputs=784, n_outputs=10, n_connections=100)

plt.figure(figsize=(10, 4))
plt.subplot(121).set_title('weights')
plt.imshow(brain_mnist.weights)
plt.subplot(122).set_title('mask')
plt.imshow(brain_mnist.mask)
plt.show()
plt.imshow(brain_mnist.predict_history(x_train[0], constant_input=True)); plt.show()
# before training
plt.figure(figsize=(10, 5))
brain_mnist.initialize_neurons()
brain_mnist.initialize_weights(2030)
plt.title('without training')
plt.imshow(brain_mnist.predict_brain_history(x_train[0], 20, constant_input=True))
plt.show()

weights_before = brain_mnist.weights.reshape(-1)[brain_mnist.mask.reshape(-1)]
%%time
# training
brain_mnist.train_hebbian_brain_batch(x_train, 1000)
# Weights
plt_bins = np.arange(-1, 1.02, 0.02)
plt.figure(figsize=(12, 4))
plt.title('Weight distributions change')
plt.hist(weights_before, bins=plt_bins, label='before training', alpha=0.7)
plt.hist(brain_mnist.weights.reshape(-1)[brain_mnist.mask.reshape(-1)], bins=plt_bins, label='after training', alpha=0.7)
plt.xlabel('weight'); plt.ylabel('N weights')
plt.legend(); plt.grid(); plt.show()
plt.imshow(brain_mnist.predict_history(x_train[0], constant_input=True)); plt.show()
plt.imshow(brain_mnist.predict_history(x_train[333], constant_input=True)); plt.show()
plt.imshow(brain_mnist.predict_history(x_train[333], constant_input=False)); plt.show()
