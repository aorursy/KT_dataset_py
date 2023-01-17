import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
tf.__version__
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
type(test_images)
type(test_labels)
# create an array with labels corresponding to the right integer values

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
%matplotlib inline



plt.figure()

plt.imshow(test_images[0], cmap=plt.cm.binary)#imshow is used to display pixel images

plt.colorbar()

plt.show()
print(train_images.shape)

print(test_images.shape)
test_images = test_images/255.0

train_images = train_images/255.0
%matplotlib inline



plt.figure()

plt.imshow(test_images[0], cmap=plt.cm.binary)

plt.colorbar()

plt.show()
%matplotlib inline



plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.xlabel(class_names[test_labels[i]])

    plt.imshow(test_images[i], cmap=plt.cm.binary)

plt.show()

    
model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),

    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)

])
#compile model

model.compile(optimizer="adam",

             loss="sparse_categorical_crossentropy",# We typically use categorical_entropy for one-hot and the sparse for softmax

             metrics=["accuracy"])
#train the model on 10 epochs

model.fit(train_images, train_labels, epochs=10)
model.evaluate(test_images, test_labels)
prediction = model.predict(np.array([test_images[0]]))
print("Prediction = {}, Actual Label = {}".format(class_names[np.argmax(prediction[0])],class_names[test_labels[0]]))
# The only difference is in the model.

cnn_model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), padding="same",

                           activation=tf.nn.relu,input_shape=(28,28,1)),

    tf.keras.layers.MaxPool2D((2,2), strides=2),

    tf.keras.layers.Conv2D(64,(3,3), padding="same",activation=tf.nn.relu),

    tf.keras.layers.MaxPool2D((2,2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),

    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)

])
cnn_model.compile(optimizer="adam",

             loss="sparse_categorical_crossentropy",# We typically use categorical_entropy for one-hot and the sparse for softmax

             metrics=["accuracy"])
# A very necessary step for our CNN input layer

train_images = train_images.reshape(60000,28,28,1)

test_images = test_images.reshape(10000,28,28,1)
#train the model on 10 epochs

cnn_model.fit(train_images, train_labels, epochs=10, batch_size=64)

print("Training complete")
cnn_model.evaluate(test_images, test_labels)