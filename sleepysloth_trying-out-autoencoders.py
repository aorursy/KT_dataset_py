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
import matplotlib.pyplot as plt

import tensorflow as tf



from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Model
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.

x_test = x_test.astype('float32') / 255.
print (x_train.shape)

print (x_test.shape)
latent_dim = 64 



class Autoencoder(Model):

    def __init__(self, latent_dim):

        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim   

        self.encoder = tf.keras.Sequential([layers.Flatten(),

                                            layers.Dense(latent_dim, activation='relu')])

        self.decoder = tf.keras.Sequential([layers.Dense(784, activation='sigmoid'),

                                            layers.Reshape((28, 28))])



    def call(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

    

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())



autoencoder.fit(x_train, x_train,

                epochs=10,

                shuffle=True,

                validation_data=(x_test, x_test))



print(autoencoder.summary())



encoded_imgs = autoencoder.encoder(x_test).numpy()

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()



n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i])

    plt.title("original")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i])

    plt.title("reconstructed")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
# 3 hidden layers and 128 units

latent_dim = 128



class Autoencoder(Model):

    def __init__(self, latent_dim):

        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([layers.Flatten(), 

                                            layers.Dense(784, activation='relu'),

                                            layers.Dense(1024, activation='relu'), 

                                            layers.Dense(latent_dim, activation='relu')])

        self.decoder = tf.keras.Sequential([layers.Dense(1024, activation='sigmoid'),

                                            layers.Dense(784, activation='sigmoid'),

                                            layers.Reshape((28, 28))])

                                           

    def call(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

    

autoencoder = Autoencoder(latent_dim)



autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())



autoencoder.fit(x_train, x_train,

                epochs=10,

                shuffle=True,

                validation_data=(x_test, x_test))



print(autoencoder.summary())

encoded_imgs = autoencoder.encoder(x_test).numpy()

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()



n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i])

    plt.title("original")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i])

    plt.title("reconstructed")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
# 5 hidden layers and 256 units

latent_dim = 256



class Autoencoder(Model):

    def __init__(self, latent_dim):

        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([layers.Flatten(), 

                                            layers.Dense(784, activation='relu'),

                                            layers.Dense(1024, activation='relu'),

                                            layers.Dense(2048, activation='relu'),

                                            layers.Dense(4096, activation='relu'),

                                            layers.Dense(latent_dim, activation='relu')])

        self.decoder = tf.keras.Sequential([layers.Dense(1024, activation='sigmoid'),

                                            layers.Dense(4096, activation='sigmoid'),

                                            layers.Dense(2048, activation='sigmoid'),

                                            layers.Dense(784, activation='sigmoid'),

                                            layers.Reshape((28, 28))])

                                           

    def call(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

    

autoencoder = Autoencoder(latent_dim)



autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())



autoencoder.fit(x_train, x_train,

                epochs=10,

                shuffle=True,

                validation_data=(x_test, x_test))



print(autoencoder.summary())

encoded_imgs = autoencoder.encoder(x_test).numpy()

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()



n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i])

    plt.title("original")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i])

    plt.title("reconstructed")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
from tensorflow.keras import regularizers



# 3 hidden layers and 128 units

latent_dim = 128



class Autoencoder(Model):

    def __init__(self, latent_dim):

        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([layers.Flatten(), 

                                            layers.Dense(784, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(1024, activation='relu', activity_regularizer=regularizers.l1(10e-5)), 

                                            layers.Dense(latent_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))])

        self.decoder = tf.keras.Sequential([layers.Dense(1024, activation='sigmoid',activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(784, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Reshape((28, 28))])

                                           

    def call(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

    

autoencoder = Autoencoder(latent_dim)



autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())



autoencoder.fit(x_train, x_train,

                epochs=10,

                shuffle=True,

                validation_data=(x_test, x_test))



print(autoencoder.summary())

encoded_imgs = autoencoder.encoder(x_test).numpy()

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()



n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i])

    plt.title("original")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i])

    plt.title("reconstructed")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()

from tensorflow.keras import regularizers



# 5 hidden layers and 256 units

latent_dim = 256



class Autoencoder(Model):

    def __init__(self, latent_dim):

        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([layers.Flatten(), 

                                            layers.Dense(784, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(1024, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(2048, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(4096, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(latent_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))])

        self.decoder = tf.keras.Sequential([layers.Dense(4096, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(2048, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(1024, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(784, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Reshape((28, 28))])

                                           

    def call(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

    

autoencoder = Autoencoder(latent_dim)



autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())



autoencoder.fit(x_train, x_train,

                epochs=10,

                shuffle=True,

                validation_data=(x_test, x_test))



print(autoencoder.summary())

encoded_imgs = autoencoder.encoder(x_test).numpy()

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()



n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i])

    plt.title("original")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i])

    plt.title("reconstructed")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
from tensorflow.keras import regularizers



# 7 hidden layers and 128 units

latent_dim = 128 # We do not increase the size of the hidden unit as the model might learn identity mapping.



class Autoencoder(Model):

    def __init__(self, latent_dim):

        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([layers.Flatten(), 

                                            layers.Dense(784, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(1024, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(2048, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(4096, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(4096, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(8192, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(latent_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))])

        self.decoder = tf.keras.Sequential([layers.Dense(8192, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(4096, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(4096, activation='relu', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(2048, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(1024, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Dense(784, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5)),

                                            layers.Reshape((28, 28))])

                                           

    def call(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

    

autoencoder = Autoencoder(latent_dim)



autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())



autoencoder.fit(x_train, x_train,

                epochs=10,

                shuffle=True,

                validation_data=(x_test, x_test))



print(autoencoder.summary())

encoded_imgs = autoencoder.encoder(x_test).numpy()

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()



n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i])

    plt.title("original")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i])

    plt.title("reconstructed")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
# Add noise to train set.

noise_train = np.reshape(np.random.normal(0,1, 60000*28*28), (60000, 28, 28))

noise_test = np.reshape(np.random.normal(0,1, 10000*28*28), (10000, 28, 28))

x_train_noisy = x_train + noise_train

x_test_noisy = x_test + noise_test



# We will use the model which has performed best up until now.

# 3 hidden layers and 128 units

latent_dim = 1000



class Autoencoder(Model):

    def __init__(self, latent_dim):

        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([layers.Flatten(), 

                                            layers.Dense(784, activation='relu'),

                                            layers.Dense(1024, activation='relu'), 

                                            layers.Dense(latent_dim, activation='relu')])

        self.decoder = tf.keras.Sequential([layers.Dense(1024, activation='sigmoid'),

                                            layers.Dense(784, activation='sigmoid'),

                                            layers.Reshape((28, 28))])

                                           

    def call(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

    

autoencoder = Autoencoder(latent_dim)



autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())



autoencoder.fit(x_train_noisy, x_train,

                epochs=10,

                shuffle=True,

                validation_data=(x_test_noisy, x_test))



print(autoencoder.summary())

encoded_imgs = autoencoder.encoder(x_test).numpy()

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()



n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i])

    plt.title("original")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i])

    plt.title("reconstructed")

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()