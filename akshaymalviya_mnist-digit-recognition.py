from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 

import pandas as pd 

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from keras.callbacks import callbacks

from keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Load the data

df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')
#Visualizing the shape of train data

df_train.shape
#Visualizing the shape of test data

df_test.shape
df_train.head()
df_test.head()
height = 28

width = 28

classes =10
#Normalization of data

x_train = df_train.drop(['label'], axis = 1).to_numpy().reshape(42000, 28, 28, 1).astype('float32')

x_train /= 255.

y_train = tf.keras.utils.to_categorical(df_train['label'], classes)



x_test = df_test.to_numpy().reshape(28000, 28, 28, 1).astype('float32')

x_test /= 255.
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)
#Visualization of image

g = plt.imshow(x_train[6,:,:,0])

print("Output = " +str(y_train[6,:]))
g = plt.imshow(x_test[7,:,:,0])
#Create the Convolution Model 

model =  models.Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size =(3, 3), activation = 'relu', input_shape = (28, 28, 1), padding = 'same'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(filters = 64, kernel_size =(3, 3), activation = 'relu', input_shape = (28, 28, 1), padding = 'same'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.3))                            

model.add(layers.Conv2D(filters = 64, kernel_size =(3, 3), activation = 'relu', input_shape = (28, 28, 1), padding = 'same'))

model.summary()
model.add(layers.Flatten())

model.add(layers.Dense(64, activation = 'relu'))

model.add(layers.Dropout(0.3))

model.add(layers.Dense(10, activation = 'softmax'))
model.summary()
#Compile the Model

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

  """Stop training when the loss is at its min, i.e. the loss stops decreasing.



  Arguments:

      patience: Number of epochs to wait after min has been hit. After this

      number of no improvement, training stops.

  """



  def __init__(self, patience=0):

    super(EarlyStoppingAtMinLoss, self).__init__()



    self.patience = patience



    # best_weights to store the weights at which the minimum loss occurs.

    self.best_weights = None



  def on_train_begin(self, logs=None):

    # The number of epoch it has waited when loss is no longer minimum.

    self.wait = 0

    # The epoch the training stops at.

    self.stopped_epoch = 0

    # Initialize the best as infinity.

    self.best = np.Inf



  def on_epoch_end(self, epoch, logs=None):

    current = logs.get('loss')

    if np.less(current, self.best):

      self.best = current

      self.wait = 0

      # Record the best weights if current results is better (less).

      self.best_weights = self.model.get_weights()

    else:

      self.wait += 1

      if self.wait >= self.patience:

        self.stopped_epoch = epoch

        self.model.stop_training = True

        print('Restoring model weights from the end of the best epoch.')

        self.model.set_weights(self.best_weights)



  def on_train_end(self, logs=None):

    if self.stopped_epoch > 0:

      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
#Train the Model

history = model.fit(x_train, y_train, epochs=15, validation_split = 0.2, callbacks = [EarlyStoppingAtMinLoss()])
#Visualizing the Accuracy of Model

plt.plot(history.history['accuracy'], label='acc')

plt.plot(history.history['val_accuracy'], label = 'val_acc')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()
#Prediction on Test data

y_predict = model.predict_classes(x_test, verbose=0)

m = plt.imshow(x_test[9,:,:,0])

print("Output = " +str(y_predict[9]))
submissions=pd.DataFrame({"ImageId": list(range(1,len(y_predict)+1)),

                         "Label": y_predict})

submissions.to_csv("sub.csv", index=False, header=True)