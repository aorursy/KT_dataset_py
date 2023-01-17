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
f = open(r'../input/shipsnet.json')

dataset = json.load(f)

f.close()



dataset.keys()
#Filtering out images and labels from JSON object

x = np.array(dataset['data']).astype('uint8')

y = np.array(dataset['labels']).astype('uint8')
x.shape
#Reshaping input images in order to feed to CNN

x = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])

x.shape
y.shape
# shuffle all indexes

indexes = np.arange(4000)

np.random.shuffle(indexes)



#Shuffling Images and Labels by same shuffled index

x = x[indexes]

y = y[indexes]



#Normalization

x = x/255.0
#Mapping labels with output classes

classes = {0: 'Not a Ship',

           1: 'Ship'}
#Visualizing Images with labels

plt.figure(figsize=(20,10))

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x[i].reshape(80, 80, 3), cmap=plt.cm.binary)

    plt.xlabel(classes[y[i]])
#Spilting data into training and testing set

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
x_train.shape
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
#Creating Checkpoint for saving weights

checkpoint_path = "training_ship/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)



# Create checkpoint callback

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 

                                                 save_weights_only=True,

                                                 verbose=1)









training_ship = cnn_model((80,80,3))



#Calculating loss ,Accuracy & also optimizing it 

training_ship.compile(optimizer = tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
epochs = 30

batch_size = 256

vali_split = 0.33



# Training

import time

start_time = time.time()



trained_ship = training_ship.fit(x_train, y_train, validation_split = vali_split , epochs = epochs, callbacks = [cp_callback], batch_size = batch_size)



training_time = time.time() - start_time
!ls {checkpoint_dir}

#model.load_weights(checkpoint_path)
#Evaluating model on testing set

training_ship.evaluate(x_test, y_test)



classifications = training_ship.predict(x_test)
print('Actual Output is {}.'.format(classes[y_test[0]]))



print('Predicted Output is {}.'.format(classes[np.argmax(classifications[0])]))
#Visualizing Loss and Accuracy



mm = training_time // 60

ss = training_time % 60

print('Training {} epochs in {}:{}'.format(epochs, int(mm), round(ss, 1)))



# show the loss and accuracy

loss = trained_ship.history['loss']

val_loss = trained_ship.history['val_loss']

acc = trained_ship.history['acc']

val_acc = trained_ship.history['val_acc']



# loss plot

plt.plot(loss)

plt.plot(val_loss, 'r')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss')

plt.legend(["Training", "Validation"])



plt.show()



# accuracy plot

plt.plot(acc)

plt.plot(val_acc, 'r')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Accuracy')

plt.legend(['Training', 'Validation'], loc=4)
#Checking out accuracy score of Model



from sklearn.metrics import accuracy_score

pred = training_ship.predict(x_test)

# convert predicions from categorical back to 0...9 digits

pred_digits = np.argmax(pred, axis=1)



print('Accuracy Score for Ships in Satellite Imagery is {}'.format(accuracy_score(y_test, pred_digits)*100))