# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

import seaborn as sns

from sklearn import preprocessing

from keras.models import Model

from keras.layers import Activation, Dense

from keras.layers import Dense, Dropout, Input, ReLU

from keras.models import Model, Sequential

from keras.optimizers import Adam

import json, codecs

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sp = pd.read_csv("../input/spindledata/spindle_data.csv")

spT=sp.T

spT.head()
sp.shape
spT.info()
plt.figure(figsize=(20,10))        

plt.plot(sp['1'])

plt.title('Input Signal 1')

plt.figure(figsize=(20,10))  

plt.plot(sp['84'])

plt.title('Input Signal 84')
#Veriyi Normalize Etme İşlemi

n_scaler = preprocessing.Normalizer()

spT = n_scaler.fit_transform(spT)

print(spT.shape)
spT = pd.DataFrame(spT)

spT = spT.loc[:,:255]

spT.shape
spT.head
#%% create generator

def create_generator():

    generator = Sequential()

    generator.add(Dense(units = 256, input_dim = 100))

    

    generator.add(Dense(units = 512))

    

    generator.add(Dense(units = 512))

    

    generator.add(Dense(units = 1024))

    

    generator.add(Dense(units = 256, activation = "tanh"))

    generator.compile(loss = "binary_crossentropy",optimizer = Adam(beta_1 = 0.5))

    return generator
g = create_generator()

g.summary()
#%% dsicriminator

def create_discriminator():

    discriminator = Sequential()

    discriminator.add(Dense(units=512,input_dim = 256))

    discriminator.add(ReLU())

    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units=256))

    discriminator.add(ReLU())

    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units=128))

    discriminator.add(ReLU())

    discriminator.add(Dense(units=1, activation = "linear"))

    discriminator.compile(loss = "binary_crossentropy",metrics=['accuracy'], optimizer= Adam( beta_1=0.5))

    return discriminator
d = create_discriminator()

d.summary()
#%% gans

def create_gan(discriminator, generator):

    discriminator.trainable = False

    gan_input = Input(shape=(100,))

    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(inputs = gan_input, outputs = gan_output)

    gan.compile(loss = "binary_crossentropy", optimizer="adam")

    return gan
gan = create_gan(d,g)

gan.summary()
# %% train

epochs = 1

batch_size = 84

for e in range(epochs):

    for _ in range(batch_size):

        noise = np.random.rand(batch_size,100)

        generated_signals = g.predict(noise)

        signal_batch = spT[np.random.randint(low = 0, high = spT.shape[0],size = 256)]

        x = np.concatenate([signal_batch, generated_signals])

        y_dis = np.zeros(batch_size*2)

        y_dis[:batch_size] = 0.9

        d.trainable = True

        d.train_on_batch(x,y_dis)

        noise = np.random.rand(batch_size,100)

        y_gen = np.ones(batch_size)

        d.trainable = False

        gan.train_on_batch(noise, y_gen)

    print("epochs: ",e)
#%% save model

g.save_weights('gans_model.h5')
#%% visualize

noise= np.random.normal(loc=0, scale=0.5, size=[500, 100])

generated_signals = g.predict(noise)

generated_signalsT = generated_signals.T

generated_signalsT.shape

table = pd.DataFrame(generated_signalsT)

table.head()

table.shape
plt.figure(figsize=(20,5))        

plt.plot(table[499])

plt.title('Output Signal 500')
plt.figure(figsize=(20,5))        

plt.plot(table[0])

plt.title('Output Signal 1')
plt.figure(figsize=(20,5))

a=sp.loc[:255,'1']

plt.plot(a)

plt.title('Input Signal 1')
table.to_csv('results.csv', header = True)