

# Import TensorFlow and TensorFlow Datasets

import tensorflow as tf



from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Helper libraries

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



print(tf.__version__)

train_dataset = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv').values

test_dataset = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv').values



num_train_examples = len(train_dataset)

num_test_examples = len(test_dataset)



print("Number of training examples: {}".format(num_train_examples))

print("Number of test examples:     {}".format(num_test_examples))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

num_classes = len(class_names)


# The map function applies the normalize function to each element in the train

# and test datasets



train_dataset_x = train_dataset[:,1:] / 255

train_dataset_x = train_dataset_x.reshape((num_train_examples, 28, 28, 1))

train_dataset_y = train_dataset[:,0]



test_dataset_x = test_dataset[:,1:] / 255

test_dataset_x = test_dataset_x.reshape((num_test_examples, 28, 28, 1))

test_dataset_y = test_dataset[:,0]

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(num_classes,  activation=tf.nn.softmax)

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
BATCH_SIZE = 32



train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_dataset_x, train_dataset_y, shuffle=True, batch_size=BATCH_SIZE)



test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(test_dataset_x, test_dataset_y, shuffle=False, batch_size=BATCH_SIZE)

model.fit_generator(train_generator, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
test_loss, test_accuracy = model.evaluate(test_generator, steps=math.ceil(num_test_examples/32))

print('Accuracy on test dataset:', test_accuracy)
predictions = model.predict(test_generator)
predictions.shape

predictions[0]
np.argmax(predictions[6])
test_dataset_y[6]
def plot_image(i, predictions_array, true_labels, images):

  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  

  plt.imshow(img[...,0], cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:

    color = 'blue'

  else:

    color = 'red'

  

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

  predictions_array, true_label = predictions_array[i], true_label[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  thisplot = plt.bar(range(10), predictions_array, color="#777777")

  plt.ylim([0, 1]) 

  predicted_label = np.argmax(predictions_array)

  

  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('blue')
i = 0

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions, test_dataset_y, test_dataset_x)

plt.subplot(1,2,2)

plot_value_array(i, predictions, test_dataset_y)
i = 12

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions, test_dataset_y, test_dataset_x)

plt.subplot(1,2,2)

plot_value_array(i, predictions, test_dataset_y)
# Plot the first X test images, their predicted label, and the true label

# Color correct predictions in blue, incorrect predictions in red

num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(i, predictions, test_dataset_y, test_dataset_x)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(i, predictions, test_dataset_y)

# Grab an image from the test dataset

img = test_dataset_x[2,:]



print(img.shape)
# Add the image to a batch where it's the only member.

img = np.array([img])



print(img.shape)
predictions_single = model.predict(img)



print(predictions_single)
plot_value_array(0, predictions_single, test_dataset_y)

_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])