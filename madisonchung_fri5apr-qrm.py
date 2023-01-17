!pip install -U tensorflow_datasets
from __future__ import absolute_import, division, print_function, unicode_literals





# Import TensorFlow and TensorFlow Datasets

import tensorflow as tf

import tensorflow_datasets as tfds

tf.logging.set_verbosity(tf.logging.ERROR)



# Helper libraries

import math

import numpy as np

import matplotlib.pyplot as plt



# Improve progress bar display

import tqdm

import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm





print(tf.__version__)



# This will go away in the future.

# If this gives an error, you might be running TensorFlow 2 or above

# If so, the just comment out this line and run this cell again

tf.enable_eager_execution()  
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
num_train_examples = metadata.splits['train'].num_examples

num_test_examples = metadata.splits['test'].num_examples

print("Number of training examples: {}".format(num_train_examples))

print("Number of test examples:     {}".format(num_test_examples))
def normalize(images, labels):

    images = tf.cast(images, tf.float32)

    images /= 255

    return images, labels



# The map function applies the normalize function to each element in the train

# and test datasets

train_dataset =  train_dataset.map(normalize)

test_dataset  =  test_dataset.map(normalize)
# Take a single image, and remove the color dimension by reshaping

for image, label in test_dataset.take(6):

    #break

    image = image.numpy().reshape((28,28))



# Plot the image - voila a piece of fashion clothing

plt.figure()

plt.imshow(image, cmap=plt.cm.binary)

plt.colorbar()

plt.grid(False)

plt.show()
plt.figure(figsize=(10,10))

i = 0

for (image, label) in test_dataset.take(25):

    image = image.numpy().reshape((28,28))

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(image, cmap=plt.cm.binary)

    plt.xlabel(class_names[label])

    i += 1

plt.show()
model = tf.keras.Sequential([

    

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    

    # Since the number of the class is 10, the output layer has 10 nodes

    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)



])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
BATCH_SIZE = 32



train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)



# while doing test, we don't need to 'repeat()' like the above,

# since we do not need to train on the test data but want to see the performance of the model we have trained.

test_dataset = test_dataset.batch(BATCH_SIZE)
# Also, we can see the how the model has been constructed. 

# how many layers are working and how many parameters we have to figure out, and so forth

model.summary()
model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
# Evaluate the accuracy

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))

print('Accuracy on test dataset:', test_accuracy)