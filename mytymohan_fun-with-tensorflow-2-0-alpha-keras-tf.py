!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import keras

print("The TensorFlow version installed in the Notebook is TensorFlow {}".format(tf.__version__))

print("The TensorFlow version of Keras installed in the Notebook is Keras {}".format(keras.__version__))
#Defining in and out

import numpy as np

x_in = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

y_out = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#Compiling the model

model.compile(optimizer='sgd', loss='mean_squared_error')
#Training the network

model.fit(x_in, y_out, epochs=50)
print("For the x_in of 10, the value of y_out is {}".format(model.predict([10.0])))
#Installing the datasets inbuilt in TensorFLow

!pip install -U tensorflow_datasets



#Loading the TensorFLow datasets

import tensorflow_datasets as tfds



#Plotting packages

import matplotlib.pyplot as plt



#Improve progress bar display

import tqdm

import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm
#Loading the Fashion MNIST dataset

mnist = tf.keras.datasets.fashion_mnist



#Getting training & testing datasets

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(training_images[4])

print(training_labels[4])

print(training_images[4])
#Normalizing the pixiel values

training_images  = training_images / 255.0

test_images = test_images / 255.0
model = tf.keras.Sequential([tf.keras.layers.Flatten(), #Flattening the image into numpy array

                             tf.keras.layers.Dense(128, activation=tf.nn.relu), #First hidden layer with 128 units & relu activation function

                             tf.keras.layers.Dense(10, activation=tf.nn.softmax) #Output layer with 10 units to give the probability of 10 output classes

                           ])
#Compiling the model

model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
#Training the model

model.fit(training_images, training_labels, epochs=10, batch_size=6000)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)



print(classifications[0])
print(test_labels[0])
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,

                           input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
#Data reshaping for training

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)

training_images=training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)

test_images=test_images/255.0
#Training the model

model.fit(training_images, training_labels, epochs=10, batch_size=6000)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Accuracy on test dataset:', test_accuracy)

print('Loss on the test dataset:', test_loss)