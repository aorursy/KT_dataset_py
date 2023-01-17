# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

import numpy as np

from keras import Sequential

class mycallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self,epoch,logs={}):

    if(logs.get('loss')<0.3):

      print("Reached 60%accuracy so cancelling the training")

      self.model.stop_training=True
callbacks=mycallback()

mnist=tf.keras.datasets.fashion_mnist

(training_images,training_labels),(test_images,test_labels)=mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(training_images[0])

print(training_images[0])

print(training_labels[0])





training_images=training_images/255.0

test_images=test_images/255.0



#it will be done because our pixels will be in between 0 to 255 and if we divide the above values by 255 then we will get the output in between 0 to 1 and that is called 

#normalising#Designing the model



model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),

                                 tf.keras.layers.Dense(128,activation=tf.nn.relu),

                                 tf.keras.layers.Dense(10,activation=tf.nn.softmax)])##Flatten -> its the shape of the input image

#Dense with Relu-> Its a hidden layer.

#Dense(10)->its a final layer with 10 different classes to detect.





model.compile(optimizer=tf.train.AdamOptimizer(),

             loss='sparse_categorical_crossentropy')

model.fit(training_images,training_labels,epochs=15, callbacks=[callbacks])

model.evaluate(test_images,test_labels)

classification=model.predict(test_images)

print(classification[34])

print(test_labels[34])