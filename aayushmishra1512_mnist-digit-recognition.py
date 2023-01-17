import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

import matplotlib.pyplot as plt
objects = tf.keras.datasets.mnist

(training_images,training_labels),(test_images,test_labels) = objects.load_data()
for i in range(9):

  plt.subplot(330 + 1 + i)

  plt.imshow(training_images[i])
print(training_images.shape) #printing the shape of our data - - 6000 images, each image of 28x28

print(training_images[0]) #printing how our image looks like in its matrix form, each image ranges from 0 pixel to 255 pixels
training_images = training_images / 255.0 #normalising our data and converting the values of our data to falal in the range 0 to 1

testing_images  = test_images / 255.0
#building our model

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (28,28)), #converting  our 2d image into 1d to feed it to our neural network

                                    tf.keras.layers.Dense(128, activation= 'relu'),

                                    tf.keras.layers.Dense(10,activation= tf.nn.softmax)]) #our output lies between 0-9 hence our output has 10 layers
model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=50)
print(model.evaluate(test_images,test_labels))
plt.imshow(testing_images[100])

pred = model.predict(testing_images)

print(np.argmax(pred[100]))
model.save('digit.h5')