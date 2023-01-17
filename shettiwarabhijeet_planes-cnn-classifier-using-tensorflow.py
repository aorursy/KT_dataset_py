from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

import pandas as pd

import numpy as np

import time

import os

import matplotlib.pyplot as plt

import json, sys, random

from sklearn.metrics import accuracy_score
tf.__version__
# Choose dataset - json object

f_plane = open(r'../input/planesnet.json')

plane_dataset = json.load(f_plane)

f_plane.close()



plane_dataset.keys()
#Filtering images and labels from JSON object

plane_x = np.array(plane_dataset['data']).astype('uint8')

plane_y = np.array(plane_dataset['labels']).astype('uint8')
plane_x.shape
#Reshaping input images in order to feed to CNN

plane_x = plane_x.reshape([-1, 3, 20, 20]).transpose([0,2,3,1])

plane_x.shape
plane_y.shape
# shuffle all indexes

shuffle_index = np.arange(32000)

np.random.shuffle(shuffle_index)



#Shuffling Images and Labels by same shuffled index

plane_x = plane_x[shuffle_index]

plane_y = plane_y[shuffle_index]



#Normalization

plane_x = plane_x / 255.0
#Mapping labels with output classes

plane_classes = {0: 'No Plane',

                 1: 'Plane'}
#Visualizing Images with labels

plt.figure(figsize=(20,10))

for i in range(8):

    plt.subplot(1,8,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(plane_x[i].reshape(20, 20, 3), cmap=plt.cm.binary)

    plt.xlabel(plane_classes[plane_y[i]])
#Spilting data into training and testing set

plane_x_train, plane_x_test, plane_y_train, plane_y_test = train_test_split(plane_x,plane_y,test_size = 0.2)
#Creating CNN model



def cnn_model(inputshape):

  model = tf.keras.Sequential([

      keras.layers.Conv2D(32, (5,5), activation = 'relu', input_shape = inputshape , padding = 'same'),

      keras.layers.Conv2D(64,(5,5), activation = 'relu'),

      keras.layers.MaxPooling2D(2,2),

      keras.layers.Dropout(0.25),

      keras.layers.Conv2D(128,(5,5), activation = 'relu'),

      keras.layers.MaxPooling2D(2,2),

      keras.layers.Dropout(0.25),

      keras.layers.Flatten(),

      keras.layers.Dense(512, activation = tf.nn.relu),

      keras.layers.Dropout(0.35),

      keras.layers.Dense(128, activation = tf.nn.relu),

      keras.layers.Dropout(0.35),

      keras.layers.Dense(2, activation = tf.nn.softmax)

  ])

  model.summary()

  return(model)
#Creating Checkpoint

checkpoint_path1 = "training_plane/cp.ckpt"

checkpoint_dir1 = os.path.dirname(checkpoint_path1)



# Create checkpoint callback

cp_callback1 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path1, 

                                                 save_weights_only=True,

                                                 verbose=1)



training_plane = cnn_model((20,20,3))



#Calculating loss ,Accuracy & also optimizing it 

training_plane.compile(optimizer = tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])



epochs1 = 30

batch_size1 = 256

vali_split1 = 0.33





# Training

import time

start_time_plane = time.time()



trained_plane = training_plane.fit(plane_x_train , plane_y_train, validation_split = vali_split1 , epochs = epochs1, callbacks = [cp_callback1], batch_size = batch_size1)



training_time_plane = time.time() - start_time_plane



!ls {checkpoint_dir1}

#model.load_weights(checkpoint_path)
#Evaluating model on testing set

training_plane.evaluate(plane_x_test, plane_y_test)



classification_plane = training_plane.predict(plane_x_test)
#classes(classifications[0])



print('Actual Output is {}.'.format(plane_classes[plane_y[40]]))



print('Predicted Output is {}.'.format(plane_classes[np.argmax(classification_plane[40])]))
#Visualizing Loss and Accuracy



mm_plane = training_time_plane // 60

ss_plane = training_time_plane % 60

print('Training {} epochs in {}:{}'.format(epochs1, int(mm_plane), round(ss_plane, 1)))



# show the loss and accuracy

loss_plane = trained_plane.history['loss']

val_loss_plane = trained_plane.history['val_loss']

acc_plane = trained_plane.history['acc']

val_acc_plane = trained_plane.history['val_acc']



# loss plot

plt.plot(loss_plane)

plt.plot(val_loss_plane, 'r')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss')

plt.legend(["Training", "Validation"])



plt.show()



# accuracy plot

plt.plot(acc_plane)

plt.plot(val_acc_plane, 'r')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Accuracy')

plt.legend(['Training', 'Validation'], loc=4)
#Checking out accuracy score of Model

pred_plane = training_plane.predict(plane_x_test)

# convert predicions from categorical back to 0...9 digits

pred_max = np.argmax(pred_plane, axis=1)



print('Accuracy Score for Planes in Satellite Imagery is {}'.format(accuracy_score(plane_y_test, pred_max)*100))