# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Activation, Dense, Input

from tensorflow.keras.optimizers import Adam, SGD

import tensorflow.keras.initializers

import tensorflow.keras

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

plt.style.use('seaborn-whitegrid')

plt.rcParams['figure.dpi'] = 200

import random

import math

from datetime import datetime

from mpl_toolkits.mplot3d import Axes3D

from numpy import *

import pylab as p

import mpl_toolkits.mplot3d.axes3d as p3

random.seed()
#### Miscellaneous functions

def sigmoid(z):

    """The sigmoid function."""

    return 1.0/(1.0+np.exp(-z))



def sigmoid_prime(z):

    """Derivative of the sigmoid function."""

    return sigmoid(z)*(1-sigmoid(z))



def normalized_sigmoid_fkt(a, b, x):

   '''

   Returns array of a horizontal mirrored normalized sigmoid function

   output between 0 and 1

   Function parameters a = center; b = width

   '''

   s= 1/(1+np.exp(b*(x-a)))

   return 1*(s-min(s))/(max(s)-min(s)) # normalize function to 0-1



def logistic(x, minimum, maximum, slope, ec50):

    return maximum + (minimum-maximum)/(1 + (x/ec50)**slope)
array_len = 10000

x = np.random.sample(array_len)

x = x*10 - 5

y = np.empty(array_len)



# discontinuous function:

#for i in range(array_len):

#    if (x[i] < 0.5):

#        y[i] = 0

#    elif (x[i] < 1000):

#        y[i] = 1

#    else:

#         y[i] =0



# Cubic function:

y = 0.25*(x**3) + 0.75*(x**2) - 1.5*x - 2

print ("Done.")
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15)

print("Done.")
model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(1,)))

model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(lr=0.001), loss='mean_absolute_error')

history_section1 = model.fit(X_train, Y_train, epochs=300, validation_split=0.2, batch_size=100, verbose=0)

print ("Done.")

print ("Last few loss values:")

print (history_section1.history['loss'][-5:])
Y_predict = model.predict(X_test)

model.evaluate(X_test, Y_test)
plt.style.use('seaborn-whitegrid')

plt.rcParams['figure.dpi'] = 200

plt.plot(X_train, Y_train, linestyle='none', marker='.', color='blue', label='Training data')

plt.plot(X_test, Y_test, linestyle='none', marker='.', color='cyan', label='Test data')

plt.plot(X_test, Y_predict, linestyle='none', marker='x', color='red', label='Prediction')

plt.legend(loc='upper left')

plt.show()
print(model.get_weights())
array_len = 10000

range_width = 1000

x2 = np.random.random_sample((array_len,))



ec501 = 0.75

slope1 = -20

minimum1 = 0

maximum1 = 1



ec502 = 0.25

slope2 = 10

minimum2 = 0

maximum2 = 1

y2 = np.empty(array_len)



for i in range(array_len):

    #y2[i] =logistic(x2[i], minimum, maximum, slope, ec50)

    y2[i] = logistic(x2[i], minimum1, maximum1, slope1, ec501) + logistic(x2[i], minimum2, maximum2, slope2, ec502) -1        

print ("Done.")
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x2, y2, test_size=0.15)

print("Done.")
model2 = Sequential()

model2.add(Dense(2, activation='sigmoid', input_shape=(1,)))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')

history_section2 = model2.fit(X_train2, Y_train2, validation_split = 0.2, batch_size=100, epochs=300, verbose=0)

print ("Note: this does not finde the global minimum of the loss function 100% of the time. may have to run more than once.")

print ("Done.")

print ("Last few loss values:")

print (history_section2.history['loss'][-5:])
Y_predict2 = model2.predict(X_test2)

model2.evaluate(X_test2, Y_test2)
plt.style.use('seaborn-whitegrid')

plt.rcParams['figure.dpi'] = 200

plt.plot(X_train2, Y_train2, linestyle='none', marker='.', color='blue', label='Training data')

plt.plot(X_test2, Y_test2, linestyle='none', marker='.', color='cyan', label='Test data')

plt.plot(X_test2, Y_predict2, linestyle='none', marker='x', color='red', label='Prediction')

plt.legend(loc='upper left')

plt.show()
for layer in model2.layers:

    print(layer.get_config())

    print(layer.get_weights())
array_len2 = 100

range_width2 = 1000



first_input = np.empty(array_len2)

second_input = np.empty(array_len2)

for i in range(array_len2):

  first_input[i] = random.randint(0, range_width2)

  second_input[i] = random.randint(0, range_width2)



x2d = np.vstack((first_input, second_input)).T

x1 = first_input*2 + second_input*3 + 4

x2 = first_input*5 - second_input*6 - 7

y2d = x1*8+ x2*9 + 10

print("Done.")
#training_frac = 0.85

#train_max_index = math.floor(array_len * training_frac)

#X_train = x[:train_max_index,:]

#Y_train = y[:train_max_index]

#X_test = x[train_max_index:,:]

#Y_test = y[train_max_index:]



X_train3, X_test3, Y_train3, Y_test3 = train_test_split(x2d, y2d, test_size=0.15)

print("Done.")
model = Sequential()

model.add(Dense(5, activation='relu', input_shape=(2,)))

model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(lr=0.01), loss='mse')

history_section3 = model.fit(X_train3, Y_train3, validation_split = 0.2, batch_size=10, epochs=750, verbose=0)

print ("Note: this does not finde the global minimum of the loss function 100% of the time. may have to run more than once.")

print ("Done.")

print ("Last few loss values:")

print (history_section3.history['loss'][-5:])
Y_predict3 = model.predict(X_test3)

model.evaluate(X_test3, Y_test3)
plt.plot(X_test3[:,0], Y_test3, linestyle='none', marker='.', color='green', label='Test data')

plt.plot(X_test3[:,0], Y_predict3, linestyle='none', marker='x', color='red', label='Prediction')

plt.legend(loc='upper left')

plt.show()
plt.plot(X_test3[:,1], Y_test3, linestyle='none', marker='.', color='green', label='Test data')

plt.plot(X_test3[:,1], Y_predict3, linestyle='none', marker='x', color='red', label='Prediction')

plt.legend(loc='upper left')

plt.show()
for layer in model.layers:

    print(layer.get_weights())



#blob1 = np.dot(X_test[0,:], model.layers[0].get_weights()[0][0]) + model.layers[0].get_weights()[1][0]

#blob1 = np.dot(X_test[0,:], model.layers[0].get_weights()[0][1]) + model.layers[0].get_weights()[1][1]

#blob3 = blob1*model.layers[1].get_weights()[0][0] + blob2*model.layers[1].get_weights()[0][1] + model.layers[1].get_weights()[1]

#print(blob3)
fig=p.figure()

ax = p3.Axes3D(fig)

ax.scatter(xs=X_test3[:,0], ys=X_test3[:,1], zs=Y_test3, zdir='z', s=20, c=None, depthshade=True, color='blue')

ax.scatter(xs=X_test3[:,0], ys=X_test3[:,1], zs=Y_predict3, zdir='z', s=20, c=None, depthshade=True, color='red')

#ax.legend()



ax.set_xlabel('$x1$', fontsize=20)

ax.set_ylabel('$x2$', fontsize=20)

#ax.yaxis._axinfo['label']['space_factor'] = 3.0

# set z ticks and labels

#ax.set_zticks([-2, 0, 2])

# change fontsize

#for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)

# disable auto rotation

ax.xaxis.set_rotate_label(False)

ax.yaxis.set_rotate_label(False)

ax.zaxis.set_rotate_label(False) 

ax.set_zlabel('$y$', fontsize=30, rotation = 0)
histories = []

num_trials = 40

for h in range(num_trials):

    print(h)

    model = Sequential()

    model.add(Dense(5, activation='relu', input_shape=(2,)))

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(lr=0.01), metrics=['accuracy'], loss='mse')

    history = model.fit(X_train3, Y_train3, validation_split = 0.2, batch_size=10, epochs=2000, verbose=0)

    histories.append(history)
plt.style.use('seaborn-whitegrid')

plt.rcParams['figure.dpi'] = 200

for j in range(num_trials):

    history = histories[j]

    plt.plot(history.history['loss'])

    #plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.yscale('log')

plt.xlabel('epoch')

#plt.legend(['train', 'test'], loc='upper left')

plt.show()
at1000 = []

at_1000_sorted = []

for j in range(num_trials):

    at1000.append(histories[j].history['loss'][1000])

    at_1000_sorted.append(histories[j].history['loss'][1000])

at_1000_sorted.sort()

#print(at1000)

numerals = [i for i in np.arange(num_trials)]

#print(numerals)

plt.style.use('seaborn-whitegrid')

plt.rcParams['figure.dpi'] = 200

plt.scatter(numerals, at1000, cmap='blue')

#plt.scatter(numerals, at_1000_sorted, cmap='red')

#plt.title('')

plt.ylabel('loss at epoch 1000')

plt.yscale('log')

plt.xlabel('trial number')

#plt.legend(['in original order', 'sorted'], loc='center left')

plt.show()
# Next: try to display the model fits using FacetGrid

#import seaborn as sns

#sns.set(style="ticks", color_codes=True)

#g = sns.FacetGrid(histories[0], col="loss", col_wrap=5, height=1.5)