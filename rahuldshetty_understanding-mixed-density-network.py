# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import tensorflow as tf

import math
def get_data(SAMPLE=1000):

    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, SAMPLE))).T

    r_data = np.float32(np.random.normal(size=(SAMPLE, 1))) 

    y_data = np.float32(7.0*np.sin(0.75*x_data) + 0.5*x_data + r_data)

    return x_data, y_data



def plot(x,y):

    plt.figure(figsize=(8, 8))

    plt.plot(x, y, 'ro', alpha=0.3)

    plt.show()



x,y = get_data(3500)

plot(x, y)
print(x.shape, y.shape)
import keras.backend as K



def square_loss(y, y_pred):

    loss = K.sum(K.square(y-y_pred))

    return loss 

    



def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,)),

        tf.keras.layers.Dense(1)

    ])

    model.compile(

         loss=square_loss, 

         optimizer=tf.keras.optimizers.Adam()

    )

    return model



model1 = build_model()

model1.summary()
import tensorflow as tf

class LoggerEpochEnd(tf.keras.callbacks.Callback):

    def __init__(self, display):

        self.seen = 0

        self.display = display

    

    def on_epoch_end(self, epoch, logs={}):

        self.seen += 1

        if self.seen%self.display == 0 :

            self.seen = 1

            print('Epochs: {} Loss:{}'.format(epoch, logs.get('loss')))     

            

logger = LoggerEpochEnd(2000)
# train the model1 for x,y and predict for some random sample

print("Training model...")

h1 = model1.fit(x,y, epochs=1000, verbose=0, callbacks=[logger])



x_test,_ = get_data(500)

y_pred = model1.predict(x_test)



def plot_multiple(x_data,y_data,x_test,y_test):

    plt.plot(x_data, y_data, 'ro', x_test, y_test, 'bo', alpha=0.3)

    plt.show()

    

plot_multiple(x,y, x_test, y_pred)
# plot the new graph

plot(y, x)
# create a new model and train it on this graph

print("Training model...")

model2 = build_model()

h2 = model2.fit(y, x, epochs=2000, verbose=0)



_,y_test = get_data(500)

x_pred = model2.predict(y_test)



plot_multiple(y,x, y_test, x_pred)
import tensorflow as tf

import tensorflow.keras.backend as K

HIDDEN = 20

KMIX = 24

NOUT = KMIX * 3 



def build_mdn_model():

    inputs = tf.keras.layers.Input(shape=(1,))

    outputs = tf.keras.layers.Dense(NOUT, activation='tanh')(inputs)

    # input - x

    # output [PI, Std Dev, Mean]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model



model_mdn = build_mdn_model()

model_mdn.summary()
import math



#oneDivSqrt2PI = 1 / math.sqrt(2*math.pi)



def extract_bayesians(outputs):

    pi = outputs[: , : KMIX ]

    std = outputs[: , KMIX : KMIX*2 ]

    mean =  outputs[: , KMIX*2 : ]

    

    # use softmax to normalize pi into prob distribution

    max_pi = K.max(pi, axis=1, keepdims=True)

    pi = pi - max_pi

    pi = K.exp(pi)

    norm_pi = 1 / K.sum(pi, axis=1, keepdims=True)

    pi = norm_pi * pi

    

    # exponent standard deviation

    # mean remains the same

    std = K.exp(std)

    

    return pi, std, mean



def keras_normal(y, mean, std):

    # normalize distribution

    result = y - mean

    result = result * ( 1 / std )

    result = - K.square(result)/2

    result = (K.exp(result) * (1 / std) ) 

    return result

    

def get_loss_func(pi, std, mean, y):

    result = keras_normal(y, mean, std)

    result = result * pi

    result = K.sum(result, axis=1, keepdims=True)

    result = -K.log(result)

    return K.mean(result)



def mdn_loss(y_true, y_pred):

    pi, std, mean = extract_bayesians(y_pred)

    return get_loss_func(pi, std, mean, y_true)
model_mdn.compile(loss=mdn_loss, optimizer='adam')

epochs = 25000

history = model_mdn.fit(y,x, epochs = epochs, verbose = 0)
plt.plot(history.history['loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
# save the module for re-use

model_mdn.save('mdn.h5')
def get_pi_index(x, pdf):

    N = pdf.shape[0]

    accum = 0

    for i in range(N):

        accum += pdf[i]

        if accum >= x:

            return i

    print("Error sampling ensemble")

    return -1





def generate_ensemble(samples, pi, std, mean, M = 10):

    size = samples.shape[0]

    result = np.random.rand(size, M) # initialize random [0, 1]

    rn = np.random.randn(size, M) # normal matrix

    

    l_mean, l_std, idx = 0, 0 ,0

    for j in range(M):

         for i in range(size):

                idx = get_pi_index(result[i, j], pi[i])

                l_mean = mean[i, idx]

                l_std = std[i, idx]

                result[i, j] = l_mean + rn[i, j]*l_std

    return result

    

def generate_distribution(samples):

    # get the result and its bayesian data

    outputs = model_mdn.predict(samples)

    pi, std, mean = extract_bayesians(outputs)

    x_test = generate_ensemble(samples, pi, std ,mean)

    return x_test 
x_pred = generate_distribution(y_test)

plot_multiple(y,x, y_test, x_pred)