# Importing statements

import pandas as pd

import numpy as np

import os
test = pd.read_csv('https://github.com/dssahota/mpcs53120/raw/master/test.csv')



train1 = pd.read_csv('https://github.com/dssahota/mpcs53120/raw/master/train1.csv')



train2 = pd.read_csv('https://github.com/dssahota/mpcs53120/raw/master/train2.csv')



train3 = pd.read_csv('https://github.com/dssahota/mpcs53120/raw/master/train3.csv')



train = pd.concat([train1, train2, train3], ignore_index=True)
len(train.columns.values), len(test.columns.values)
train.head()
test.head()
# Training Data

X_train = train.drop(labels=['label'], axis=1)

y_train = train['label']



# Testing Data

X_test = test.drop(labels=['ID'], axis=1)

#y_test = test['label']
X_train.shape, y_train.shape, X_test.shape
#X_train.max(), X_train.min()
# Normalize the Data

# This will increase the speed of training and in some cases it increases th accuracy\

#X_train /= 255
X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
X_train[0].shape, X_test[0].shape
import matplotlib.pyplot as plt

fig = plt.figure()

for i in range(9):

  plt.subplot(3,3,i+1)

  plt.tight_layout()

  plt.imshow(X_train[i].reshape(28, 28), cmap='gray', interpolation='none')

  plt.title("Digit: {}".format(y_train[i]))

  plt.xticks([])

  plt.yticks([])

fig
# Normalize the Data

X_train = X_train / 255.0

X_test = X_test / 255.0
X_train.max(), X_train.min()
# Encoding the labels

import tensorflow as tf

#import keras



y_train = tf.keras.utils.to_categorical(y_train, 10)
from sklearn.model_selection import train_test_split

# Splitting the Data into Training and Validation set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
# Importing Statements

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Dense, Flatten



# Building the model

model = Sequential([

    

    # First convolution Layer

    Conv2D(16, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),

    BatchNormalization(),

    Conv2D(16, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu'),

    BatchNormalization(),

    MaxPool2D(pool_size=(2, 2)),

    Dropout(0.25),

    

    # Second Convolution Layer

    Conv2D(32, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu'),

    BatchNormalization(),

    Conv2D(32, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu'),

    BatchNormalization(),

    MaxPool2D(pool_size=(2, 2)),

    Dropout(0.25),

    

    

    # Flatten the Layer

    Flatten(),

    

    # Applying the Fully-Connected Layers

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(10, activation='softmax')

])



# Summary of the Model

model.summary()
# Compiling the model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training the model

history = model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1, validation_data=(X_val, y_val))
# Predicting the output of the model

y_train_pred = model.predict(X_train)

# Convert predictions classes to one hot vectors 

y_train_pred_classes = np.argmax(y_train_pred, axis=1)

# Convert validation observations to one hot vectors

y_train_true = np.argmax(y_train, axis=1)



# Predicting the output of the model

y_val_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Convert validation observations to one hot vectors

y_val_true = np.argmax(y_val, axis=1)
# Printing the Learning Curve



# Printing the accuracy curve

fig = plt.figure()

#plt.subplot(2,1,1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Training and Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='lower right')

plt.show()



# Printing the loss curve

#plt.subplot(2,1,1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Training and Validation Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper right')



plt.show()
# printing the accuracy and confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score



# printing the accuracy score of the model

train_acc = accuracy_score(y_train_true, y_train_pred_classes)

val_acc = accuracy_score(y_val_true, y_val_pred_classes)



# printing the confusion matrix of the model

#confusion_matrix(y_train, y_train_pred)

train_conf = confusion_matrix(y_train_true, y_train_pred_classes)

val_conf = confusion_matrix(y_val_true, y_val_pred_classes)
train_acc, val_acc
print(train_conf)
print(val_conf)
# Predicting the output of the Test set

y_test_pred = model.predict(X_test)



# Predicting the classes of the test set

y_test_pred_classes = np.argmax(y_train_pred, axis=1)
print(y_test_pred_classes)