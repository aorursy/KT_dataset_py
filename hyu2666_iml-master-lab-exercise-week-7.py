import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

import os



# TensorFlow

import tensorflow as tf

from tensorflow import keras



print(tf.__version__)
#Loading the Fashion MNIST Dataset

fashion_mnist = keras.datasets.fashion_mnist



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Training Data

train_images
#Class of clothing the image represents

train_labels
#Training Data Exploration

print("Dimensions of the training set: ", train_images.shape)

print("No. of labels in the training set: ",len(train_labels))
#Test Data Exploration

print("Dimensions of the test set: ", test_images.shape)

print("No. of labels in the test set: ",len(test_labels))
#Sample data item

plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
#Preprocessing

train_images = train_images / 255.0



test_images = test_images / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#Images from the training set

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
#Building the model

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
#Model Prediction

predictions = model.predict(test_images)
#Sample Prediction

predictions[0]
#Manual Validation

print("Predicted Value:", np.argmax(predictions[0]))

print("Actual Value:", test_labels[0])
#Visualizing the predicted item:

def plot_image(i, predictions_array, true_label, img):

  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  

  plt.imshow(img, cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:

    color = 'blue'

  else:

    color = 'red'

  

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)
#Prediction accuracy 

def plot_value_array(i, predictions_array, true_label):

  predictions_array, true_label = predictions_array[i], true_label[i]

  plt.grid(False)

  plt.xticks(range(10), class_names, rotation=45)

  plt.yticks([])

  thisplot = plt.bar(range(10), predictions_array, color="#777777")

  plt.ylim([0, 1]) 

  predicted_label = np.argmax(predictions_array)

 

  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('blue')
i = 0

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions, test_labels, test_images)

plt.subplot(1,2,2)

plot_value_array(i, predictions,  test_labels)

plt.show()
num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(i, predictions, test_labels, test_images)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(i, predictions, test_labels)

plt.show()