# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
#Load data

train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
#Check first 5 rows of data

train_data.head()
#Preprocess the training samples (images) 

X_train = np.array(train_data.drop(['label'], axis=1)).astype('float32')/255.0
#check shape and min, max value of of X

X_train.shape, X_train.max(), X_train.min()
#define some hyperparameters

BATCH_SIZE = 128

NUM_EPOCHS = 200

HIDDEN_ACTIVATION = tf.keras.layers.LeakyReLU(alpha=0.2)

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

GEN_OUTPUT_ACTIVATION = 'sigmoid'

z_dim = 100  

DROPOUT_RATE = 0.3
#Create generator Model

def create_generator():

    model = tf.keras.models.Sequential(name='GENERATOR')

    model.add(tf.keras.layers.Dense(units=256, input_dim=z_dim, activation=HIDDEN_ACTIVATION))

    model.add(tf.keras.layers.Dense(units=512, activation=HIDDEN_ACTIVATION))

    model.add(tf.keras.layers.Dense(units=1024, activation=HIDDEN_ACTIVATION))

    model.add(tf.keras.layers.Dense(units=784, activation=GEN_OUTPUT_ACTIVATION))

    model.compile(loss='binary_crossentropy',optimizer=OPTIMIZER, metrics=['accuracy'])

    print(model.summary())

    return model
#Create discriminator Model

def create_discriminator():

    model = tf.keras.models.Sequential(name='DISCRIMINATOR')

    model.add(tf.keras.layers.Dense(units=1024, input_dim=784, activation=HIDDEN_ACTIVATION))

    model.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    model.add(tf.keras.layers.Dense(units=512, activation=HIDDEN_ACTIVATION))

    model.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    model.add(tf.keras.layers.Dense(units=256, activation=HIDDEN_ACTIVATION))

    model.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer=OPTIMIZER, metrics=['accuracy'])

    print(model.summary())

    return model
#Create GAN

def create_gan(generator, discriminator):

    #Provide input for GAN model

    input_gan = tf.keras.layers.Input(shape=(z_dim,), name='Input_Noise')

    #x is ouput of genrator with input as input of GAN

    x = generator(input_gan)

    #discriminate output of gen

    discriminator.trainable = False

    output_gan = discriminator(x)

    model = tf.keras.models.Model(inputs = input_gan, outputs = output_gan, name='GAN')

    model.compile(loss='binary_crossentropy',optimizer=OPTIMIZER)

    print(model.summary())

    return model
#Create model instances for generator, discriminator and GAN

g = create_generator()

d = create_discriminator()

gan = create_gan(generator=g, discriminator=d)
losses = {'D': [], 'G':[]}



def plot_loss(losses):

    d_loss = [v[0] for v in losses["D"]]

    g_loss = [v for v in losses["G"]]

    

    plt.figure(figsize=(10,8))

    plt.plot(d_loss, label="Discriminator loss")

    plt.plot(g_loss, label="Generator loss")

    

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()



def plot_generated(n_ex=10, dim=(1, 10), figsize=(12, 2)):

    noise = np.random.normal(0, 1, size=(n_ex, z_dim))

    generated_images = g.predict(noise)

    generated_images = generated_images.reshape(n_ex, 28, 28)



    plt.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):

        plt.subplot(dim[0], dim[1], i+1)

        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')

        plt.axis('off')

    plt.tight_layout()

    plt.show()
def training(epochs=1, batch_size=128, plt_frq=1):

    batch_count = int(X_train.shape[0]/batch_size)

    print('Epochs:', epochs)

    print('Batch size:', batch_size)

    print('Batches per epoch:', batch_count)



    #Training  on each epoch:



    

    for e in tqdm_notebook(range(1, epochs+1)):

        if e == 1 or e%plt_frq == 0:

            print('-'*15, 'Epoch %d' % e, '-'*15)



        #train on each batch

        for _ in range(batch_count):

            #Create random noises to initiate fake samples:

            z = np.random.normal(0, 1, size=(batch_size, z_dim))

            #Generate fake examples from generator:

            x_hat =  g.predict(z)

            #Pick random real samples:

            x = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size), :]

            #Concatenate x and x_hat provide training samples for discriminator

            x_con = np.concatenate([x, x_hat])

            #Assign labels for x_con: the first half is is assigned as 0.9 (real), the second half is assigned as 0 (fake) 

            y_con = np.zeros(shape=(2*batch_size,1))

            y_con[:batch_size] = 0.9

            #Train the discriminator:

            d.trainable = True

            d_loss = d.train_on_batch(x_con, y_con)

            #Train GAN with z and fake label y_hat to generate better x_hat (or only generator, actually)

            d.trainable = False

            z = np.random.normal(0, 1, size=(batch_size, z_dim))

            y_hat = np.ones(shape=(batch_size, 1))

            g_loss = gan.train_on_batch(z, y_hat)

        #append loss after each epoch

        losses['D'].append(d_loss)

        losses['G'].append(g_loss)

        #Visualize generated pics after each 20 epochs

        if e == 1 or e%plt_frq == 0:

            plot_generated()

    





            



int(X_train.shape[0]//BATCH_SIZE)
model = training(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, plt_frq=20)
#plot losses after training

plot_loss(losses)
#Generate some picture

noise = np.random.normal(0,1, size=(1,100))

plt.imshow(g.predict(noise).reshape(28, 28), cmap='gray_r')