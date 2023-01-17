# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
np.random.seed(11)

tf.random.set_seed(11)

batch_size = 256

max_epochs = 50

learning_rate = 0.001

momentum = 0.8

hidden_dim = 128

original_dim = 784

(X_train, _) , (X_test, _) = tf.keras.datasets.mnist.load_data()



# Divide by 255 

X_train = X_train/255.

X_test = X_test/255.



X_train = X_train.astype(np.float32)

X_test = X_test.astype(np.float32)

X_train = np.reshape(X_train, (X_train.shape[0], 784))

X_test = np.reshape(X_test, (X_test.shape[0], 784))

# Create the dataset

train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size)



for data in train_dataset.take(1):

    print(data.shape)

# Create Encoder class

class Encoder(tf.keras.layers.Layer):

    def __init__(self, hidden_dim):

        super(Encoder, self).__init__()

        self.hidden_layer = tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.relu)

        

    def call(self, inputs):

        activation = self.hidden_layer(inputs)

        return activation
class Decoder(tf.keras.layers.Layer):

    def __init__(self, hidden_dim, original_dim):

        super(Decoder, self).__init__()

        self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu)

        

    def call(self, encoded):

        activation = self.output_layer(encoded)

        return activation
class Autoencoder(tf.keras.Model):

    def __init__(self, hidden_dim, original_dim):

        super(Autoencoder, self).__init__()

        self.loss = []

        self.encoder = Encoder(hidden_dim=hidden_dim)

        self.decoder = Decoder(hidden_dim=hidden_dim, original_dim=original_dim)

        

    def call(self, input_features):

        encoded = self.encoder(input_features)

        reconstructed = self.decoder(encoded)

        return reconstructed
autoencoder = Autoencoder(hidden_dim=hidden_dim, original_dim=original_dim)

optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum=momentum)
# Define Loss function

def loss(predictions, labels):

    return tf.reduce_mean(tf.square(tf.subtract(predictions, labels)))
# Train the model

def train(loss, model, optimizer, inputs):

    with tf.GradientTape() as tape:

        predictions = model(inputs)

        reconstructed_error = loss(predictions, inputs)

    gradients = tape.gradient(reconstructed_error, model.trainable_variables)

    gradients = zip(gradients, model.trainable_variables)

    optimizer.apply_gradients(gradients)

    

    return reconstructed_error
def train_loop(model, optimizer, loss, dataset, epochs=20):

    for epoch in range(epochs):

        epoch_loss = 0

        for step, batch_features in enumerate(dataset):

            loss_values = train(loss, model, optimizer, batch_features)

            epoch_loss += loss_values

        model.loss.append(epoch_loss)

        print('Epoch {}/{}. Loss: {}'.format(epoch + 1, epochs, epoch_loss.numpy()))

autoencoder = Autoencoder(hidden_dim=hidden_dim, original_dim=original_dim)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

train_loop(autoencoder, optimizer, loss, train_dataset, epochs=max_epochs)
plt.plot(range(max_epochs), autoencoder.loss )

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
numbers = 10

plt.figure(figsize=(20,4))

for n in range(numbers):

    ax = plt.subplot(2, numbers, n+1)

    plt.imshow(X_test[n].reshape(28,28), cmap='gray')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    # Display Autoencoder's reconstruction output

    

    ax = plt.subplot(2, numbers, n+1+numbers)

    plt.imshow(autoencoder(X_test)[n].numpy().reshape(28, 28), cmap='gray')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

plt.show()
# Add noises to input train and test



noise = np.random.normal(loc=0.5, scale=0.5, size=X_train.shape)

X_train_noise = X_train +noise

noise = np.random.normal(loc=0.5, scale=0.5, size=X_test.shape)

X_test_noise = X_test+noise
# Create Denoising Autoencoder model and train

model = Autoencoder(hidden_dim=hidden_dim, original_dim=original_dim)



model.compile(loss='mse', optimizer='adam')



loss = model.fit(X_train_noise,

                X_train,

                validation_data=(X_test_noise, X_test),

                epochs=max_epochs,

                batch_size=batch_size)
plt.plot(range(max_epochs), loss.history['loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
number = 10  # how many digits we will display

plt.figure(figsize=(20, 4))

for index in range(number):

    # display original

    ax = plt.subplot(2, number, index + 1)

    plt.imshow(X_test_noise[index].reshape(28, 28), cmap='gray')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, number, index + 1 + number)

    plt.imshow(model(X_test_noise)[index].numpy().reshape(28, 28), cmap='gray')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()