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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, InputLayer, MaxPool2D, Dense, Input, GlobalAveragePooling2D, LeakyReLU, Dropout, Flatten, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# example of training a gan on mnist
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint

train = pd.read_csv('/kaggle/input/digit-recognition-dataset/train.csv')
test  = pd.read_csv('/kaggle/input/digit-recognition-dataset/test.csv')

train_data = train.iloc[:,1:].to_numpy() / 255.0
train_label = train.iloc[:,:1].to_numpy()

test_data  = test.iloc[:,:].to_numpy() / 255.0

X_train, X_val, Y_train, Y_val = train_test_split( train_data, train_label, test_size=0.33, random_state=42)
fig = plt.figure(figsize=(4,4))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(np.reshape(train_data[i],(28,28)))
    plt.axis('off')

plt.show()
classes = 10
epochs = 10
batch_size = 64
def digit_recognition(input_size = 784):
    model = keras.Sequential()
    model.add(Dense(512,  activation = 'relu', input_shape = (input_size, )))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(classes, activation = 'softmax'))
    
    #complile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model
model = digit_recognition()
model.summary()
model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_val, Y_val))
model.evaluate(X_val, Y_val)
fig = plt.figure(figsize=(10,10))
Y_test = np.argmax(model.predict(X_test), axis = 1)

for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(np.reshape(test_data[i],(28,28)))
    #plt.axis('off')
    plt.xlabel(Y_test[i])

plt.show()
embedding_dim = 128
def autoencoder(input_size = 784):
    #encoder
    model = keras.Sequential()
    model.add(Dense(512, activation = 'relu', input_shape = (input_size, )))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(embedding_dim, activation = 'relu'))
    
    #decoder
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(input_size, activation = 'relu'))
    
    #compile model
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    return model
auto_model = autoencoder()
train_data.shape
auto_model.fit(X_train, X_train, epochs = 50, batch_size = 64)
fig = plt.figure(figsize=(8,8))

for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(np.reshape(test_data[i],(28,28)))
    plt.axis('off')
    #plt.xlabel(Y_train[i])

plt.show()
fig = plt.figure(figsize=(8,8))
X_generate = auto_model.predict(test_data)

for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(np.reshape(X_generate[i],(28,28)))  
    plt.axis('off')

plt.show()
origin_size = 784
compress = 128
print('compression ratio with original: {}'.format((compress / origin_size) * 100))
# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
    model = keras.Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model
def define_gan(g_model, d_model):
    d_model.trainable = False
    model = keras.Sequential()
    model.add(g_model)
    model.add(d_model)
    #compile model
    model.compile(optimizer='adam', loss = 'binary_crossentropy')
    
    return model
# load and prepare mnist training images
def load_real_samples():
    return train_data.reshape((train_data.shape[0], 28, 28, 1))

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y
# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        #print(i)
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
# size of the latent space
latent_dim = 128
# number of epochs
epochs = 100
# batch size
batch_size = 256
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, epochs, batch_size)
# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
 
# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    pyplot.show()
 
# load model
model = load_model('/kaggle/working/generator_model_100.h5')
# generate images
latent_points = generate_latent_points(latent_dim,n_samples=25)
# generate images
X = model.predict(latent_points)
# plot the result
save_plot(X, 5)
