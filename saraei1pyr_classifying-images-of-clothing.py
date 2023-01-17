!pip install -U tensorflow_datasets #We use this code to download dataset (tesndorflow_datasets) in GoogleCollab notebook (https://colab.research.google.com/notebooks/welcome.ipynb#recent=true)
from __future__ import absolute_import, division, print_function





# TensorFlow and TensorFlow Datasets

import tensorflow as tf

import tensorflow_datasets as tfds

tf.logging.set_verbosity(tf.logging.ERROR)



# Another libraries

import math

import numpy as np

import matplotlib.pyplot as plt

import tqdm

import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm



print(tf.__version__)

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



train_dataset =  train_dataset.map(normalize)

test_dataset  =  test_dataset.map(normalize)
for image, label in test_dataset.take(1):

  break

image = image.numpy().reshape((28,28))



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

    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
BATCH_SIZE = 32

train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)

test_dataset = test_dataset.batch(BATCH_SIZE)
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))

print('Accuracy on test dataset:', test_accuracy)
for test_images, test_labels in test_dataset.take(1):

  test_images = test_images.numpy()

  test_labels = test_labels.numpy()

  predictions = model.predict(test_images)
predictions.shape

predictions[0]
np.argmax(predictions[0])
test_labels[0]