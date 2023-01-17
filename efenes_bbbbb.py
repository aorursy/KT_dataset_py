import numpy as np

import pandas as pd

from sklearn import preprocessing

from keras.models import Sequential, Model

from keras.layers.core import Dense, Dropout

from keras.layers import LSTM, SimpleRNN, Input

from keras.layers.embeddings import Embedding

from keras.layers import Flatten

from keras.preprocessing import sequence

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.embeddings import Embedding

import numpy as np

import matplotlib.pyplot as plt

import scipy

from scipy.stats import moment

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
def generate_data(n_samples = 84,n_dim=100):

    return np.random.normal(loc=0, scale=1, size=[n_samples, n_dim]) 
def set_model(input_dim, output_dim, hidden_dim=64,n_layers = 1,activation='linear',optimizer='adam', loss = 'binary_crossentropy'):

    model = Sequential()

    model.add(Dense(hidden_dim,input_dim=input_dim,activation=activation))

    model.add(Dropout(rate=0.3))

    for _ in range(n_layers-1):

        model.add(Dense(hidden_dim),activation=activation)

    model.add(Dense(output_dim))

  

    model.compile(loss=loss, optimizer=optimizer)

    print(model.summary())

    return model
def get_gan_network(discriminator, random_dim, generator, optimizer = 'adam'):

    discriminator.trainable = False

    gan_input = Input(shape=(random_dim,))

    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(inputs = gan_input,outputs=gan_output)

    gan.compile( loss='binary_crossentropy', optimizer=optimizer)

    return gan
NOISE_DIM = 100

DATA_DIM = 256

G_LAYERS = 1

D_LAYERS = 1
def train_gan(epochs=1,batch_size=84):

    

    batch_count = spT.shape[0]/batch_size

  

    generator = set_model(NOISE_DIM, DATA_DIM, n_layers=G_LAYERS, activation='tanh',loss = 'binary_crossentropy')

    discriminator = set_model(DATA_DIM, 1, n_layers= D_LAYERS, activation='sigmoid')

    gan = get_gan_network(discriminator, NOISE_DIM, generator, 'adam')

  

    for e in range(1,epochs+1):   

    

        # Noise is generated from a uniform distribution

        noise = np.random.normal(loc=0, scale=1, size=[batch_size, NOISE_DIM]) 

        true_batch = spT[np.random.randint(low = 0, high = spT.shape[0],size = 256)]

    

        generated_values = generator.predict(noise)

        X = np.concatenate([true_batch,generated_values])

     

        y_dis = np.zeros(2*batch_size)

    

        #One-sided label smoothing to avoid overconfidence. In GAN, if the discriminator depends on a small set of features to detect real images, 

        #the generator may just produce these features only to exploit the discriminator. 

        #The optimization may turn too greedy and produces no long term benefit.

        #To avoid the problem, we penalize the discriminator when the prediction for any real images go beyond 0.9 (D(real image)>0.9). 

        y_dis[:batch_size] = 0.9

    

        discriminator.trainable = True

        disc_history = discriminator.train_on_batch(X, y_dis)

        discriminator.trainable = False



        # Train generator

        # Noise is generated from a uniform distribution

        noise = np.random.rand(batch_size,NOISE_DIM)

        y_gen = np.zeros(batch_size)    

        gan.train_on_batch(noise, y_gen)  

 

    print("epochs: ",e)

    return generator, discriminator
generator, discriminator = train_gan()
#%% save model

generator.save_weights('gans_model.h5')
#%% visualize

noise= np.random.normal(loc=0, scale=1, size=[500, 100])

generated_signals = generator.predict(noise)

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