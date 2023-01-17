# Simple MNIST program to indentify numbers

# Goal: Try out various optimizers and models to improve accuracy

# I will be using this notebook to test models and optimizers 





import tensorflow as tf

from tensorflow import keras

from keras.datasets import mnist

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
#Get training, test data

(X_train,Y_train), (X_test,y_test) = mnist.load_data()
#output class

class_output = [0,1,2,3,4,5,6,7,8,9]
#shape of the data 

print(X_train.shape, Y_train.shape, X_test.shape, y_test.shape)
#output unique values in the training data



np.unique(Y_train)
# plot test data 



plt.figure()

plt.imshow(X_test[0])

plt.colorbar()

plt.grid(False)

plt.show()
#Scale the values from 0 to 1 



X_train = X_train /255.0 

X_test = X_test /255.0
# Plot the values 



plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train[i], cmap=plt.cm.binary)

    plt.xlabel(class_output[Y_train[i]])

plt.show()
# create a model 

num_model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])
#compile a model with optimizer 



num_model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
#train the model 

num_model.fit(X_train, Y_train, epochs=10)
#Evaluate model  

test_loss, test_acc = num_model.evaluate(X_test, y_test, verbose=2)
#make predictions



probability_num_model = tf.keras.Sequential([num_model, 

                                         tf.keras.layers.Softmax()])

num_predictions = probability_num_model.predict(X_test)
#function to plot output numbers



def num_plot(i, predictions_array, true_label, img):

  true_label, img = true_label[i], img[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])



  plt.imshow(img, cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:

    color = 'blue'

  else:

    color = 'red'



  plt.xlabel("{} {:2.0f}% ({})".format(class_output[predicted_label],

                                100*np.max(predictions_array),

                                class_output[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

  true_label = true_label[i]

  plt.grid(False)

  plt.xticks(range(10))

  plt.yticks([])

  thisplot = plt.bar(range(10), predictions_array, color="#777777")

  plt.ylim([0, 1])

  predicted_label = np.argmax(predictions_array)



  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('blue')
# Plot numbers 

num_rows = 25

num_cols = 25

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  num_plot(i, num_predictions[i], y_test, X_test)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(i, num_predictions[i],  y_test)

plt.tight_layout()

plt.show()