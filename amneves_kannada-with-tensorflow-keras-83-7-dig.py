# Just some standard python machine learning imports, we will be using keras from tensorflow

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 



from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.layers import Activation

from sklearn import metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.optimizers import RMSprop



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load the data from the csv files into pandas dataframes

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

dig = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
# Extract column 0 (the labels)

labels = train.iloc[:,0].values.astype('int32')



# Extract the rest of the columns (the images)

X_train = (train.iloc[:,1:].values).astype('float32')/255



# Extract the testing images

X_test = test.iloc[:,1:].values.astype('float32')/255



# Do the same extraction for the Dig dataset

dig_labels = dig.iloc[:,0].values.astype('float32')

dig_val = dig.iloc[:,1:].values.astype('int32')/255
# Just a little bit of array shape validation

print(X_train.shape)

print(labels.shape)



print(X_test.shape)



print(dig_val.shape)

print(dig_labels.shape)
# Reshape continuous arrays into 28 by 28 greyscale images

X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)

dig_images = dig_val.reshape(-1,28,28,1)



# One hot enconde labels

y_train = tf.keras.utils.to_categorical(labels) 



# Just a check for my mental sake heh

print("Check data")

print(labels)

print(X_train[0].shape)

print(X_test[0].shape)

print(dig_images[0].shape)
# let's vizualise a single image

fig = plt.figure()

plt.imshow(X_train[6][:,:,0], cmap='gray', interpolation='none')

plt.xticks([])

plt.yticks([])
# Define the input and output layer sizes

input_size = X_train.shape

n_logits = y_train.shape[1]



print("Input: {}".format(input_size))

print("Output: {}".format(n_logits))
epochs = 50

batch_size = 1024

#get validation data



X_train, X_val, Y_train, Yval = train_test_split(X_train, y_train, train_size = 0.90)





datagen = ImageDataGenerator(rotation_range = 10,

                           width_shift_range = 0.25,

                           height_shift_range = 0.25,

                           shear_range = 0.1,

                           zoom_range = 0.25,

                           horizontal_flip = False)



datagen.fit(X_train)
# Here we define our keras model

model = tf.keras.Sequential()



# convolutions

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=input_size[1:]))

model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same'))

model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())



# head network

model.add(tf.keras.layers.Dense(256))

model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

model.add(tf.keras.layers.Dense(256))

model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

model.add(tf.keras.layers.Dropout(0.5))

# output

model.add(tf.keras.layers.Dense(n_logits, activation='softmax'))





es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)



# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)





model.compile(loss='categorical_crossentropy',

              optimizer='Adam',

              metrics=['accuracy'])



model.summary()
# Fit the model

hist = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Yval),

                              verbose = 1, steps_per_epoch=100, callbacks=[learning_rate_reduction, es])
def plot_hist(history):

    accuracy = history.history['accuracy']

    val_accuracy = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Training accuracy')

    plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')

    plt.title('Accuracy')

    plt.legend()

    plt.show()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')

    plt.plot(epochs, val_loss, 'r', label='Test loss')

    plt.title('Loss')

    plt.legend()

    plt.show()
plot_hist(hist)
dig_pred=model.predict_classes(dig_images)

print(metrics.accuracy_score(dig_pred, dig_labels))
# generate predictions

predictions = model.predict_classes(X_test, verbose=0)

pd.DataFrame({"id": list(range(0,len(predictions))), "label": predictions}).to_csv("submission.csv", index=False, header=True)