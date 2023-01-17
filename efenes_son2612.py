from keras.layers import Dense, Dropout, Input, ReLU

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Model, Sequential

from keras.optimizers import Adam

import numpy as np

import matplotlib.pyplot as plt

from scipy import signal
import pandas as pd

pd = pd.read_csv("../input/spindledata/spindle_data.csv")

sp = pd.copy()

sp = sp.loc[:255,:]
sp=sp.T

print(sp.shape)
fs = 128

f, t, Zxx = signal.stft(sp, fs)

print(Zxx.shape)
Zxx = Zxx.reshape(Zxx.shape[0],Zxx.shape[1]*Zxx.shape[2])

print(Zxx.shape)
sp=Zxx

print(sp.shape)

print(type(sp))

print(sp.dtype)
x_train=sp
#%% create generator

def create_generator():

    generator = Sequential()

    generator.add(Dense(units = 128, input_dim = 100))

    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units = 128))

    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units = 256))

    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units = 387, activation = "linear"))

    generator.compile(loss = "binary_crossentropy", optimizer = Adam(beta_1 = 0.5))

    return generator
g = create_generator()

g.summary()
#%% dsicriminator

def create_discriminator():

    discriminator = Sequential()

    discriminator.add(Dense(units=256,input_dim = 387))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units=128))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units=64))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=1, activation = "sigmoid"))

    discriminator.compile(loss = "binary_crossentropy", optimizer= Adam(beta_1=0.5))

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

losses = {"D":[], "G":[]}

epochs = 1500

batch_size = 10

for e in range(epochs):

    for _ in range(batch_size):

        noise = np.random.normal(0,1, [batch_size,100])

        generated_images = g.predict(noise)

        image_batch = x_train[np.random.randint(low = 0, high = x_train.shape[0],size = batch_size)]

        x = np.concatenate([image_batch, generated_images])

        y_dis = np.zeros(batch_size*2)

        y_dis[:batch_size] = 0.9

        d.trainable = True

        d_loss = d.train_on_batch(x,y_dis)

        noise = np.random.normal(0,1,[batch_size,100])

        y_gen = np.ones(batch_size)

        d.trainable = False

        g_loss = gan.train_on_batch(noise, y_gen)

    # Only store losses from final

    losses["D"].append(d_loss)

    losses["G"].append(g_loss)

    if (e) % 500 == 0:

        print("epochs: ",e, "g_loss",g_loss,"d_loss",d_loss)

        noise= np.random.normal(loc=0, scale=1, size=[100, 100])

        generated = g.predict(noise)

        generated=generated.reshape(100,129,3)

        amp = 2 * np.sqrt(2)

        Zxx = np.where(np.abs(generated) >= amp/10, generated, 0)

        _, xrec = signal.istft(Zxx, fs)

        plt.figure(figsize=(20,5))        

        plt.plot(xrec[1,:])

        plt.title('Output Signal 2')
noise= np.random.normal(loc=0, scale=1, size=[500, 100])

generated = g.predict(noise)

print(generated.shape)
generated=generated.reshape(500,129,3)
amp = 2 * np.sqrt(2)

Zxx = np.where(np.abs(generated) >= amp/10, generated, 0)

_, xrec = signal.istft(Zxx, fs)
xrec.shape
plt.figure(figsize=(20,5))        

plt.plot(xrec[7,:])

plt.title('Output Signal 8')
plt.figure(figsize=(20,5))        

plt.plot(pd.loc[:255,'7'])

plt.title('Input Signal 7')
plt.plot(losses["D"])
plt.plot(losses["G"])