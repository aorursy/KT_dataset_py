%matplotlib inline



import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os

import numpy as np

from matplotlib import pyplot as plt



# Load modules.

import pickle

import random

import csv

from training_example import training
# Load data.

cifar10 = pickle.load(open("cifar10_data", "rb"), encoding="bytes")

X_train = cifar10['X_train'].astype(np.float32)

y_train = cifar10['y_train'].astype(np.int64)

X_test  = cifar10['X_test'].astype(np.float32)

label_names = cifar10['label_names']



# Partition training data into training and validation sets.

X_val = X_train[:1000]

y_val = y_train[:1000]

X_train = X_train[1000:]

y_train = y_train[1000:]



# Mean subtract the data.

mean_image = np.mean(X_train, axis=0)

X_train -= mean_image

X_val -= mean_image

X_test -= mean_image



# normalize by the standard deviation of the data

std_image = np.std(X_train,axis=0)

X_train /= std_image

X_val /= std_image

X_test /= std_image



# Convert class vectors to binary class matrices.

y_train = keras.utils.to_categorical(y_train, 10)

y_val = keras.utils.to_categorical(y_val, 10)
# Visualize a random subset of labeled images from the training set.

idx = random.sample(range(len(y_train)), 25)

plt.figure(figsize=(10,10))

for n in np.arange(25):

    plt.subplot(5, 5, n+1)

    plt.imshow((X_train[idx[n]]*std_image+mean_image)/255)

    plt.xticks([])

    plt.yticks([])

    plt.title(label_names[np.where(y_train[idx[n]])[0][0]])
# in the Keras package, you can specify a model as "sequential": it then adds each layer on top of the other layers

# You need to specify the size of the inputs to the first layer, but after that the Keras internals will

# calculate the size of the data for you



model = Sequential()



# immediately flatten the input

model.add(Flatten(input_shape=X_train.shape[1:]))



# hidden layer

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))



# hidden layer

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))



# output layer

model.add(Dense(10,activation='softmax')) # output of the network



# Let's train the model using nAdam

model.compile(loss='categorical_crossentropy', # probability should be high for the correct answer and low for the wrong answers

              optimizer='nadam', # Adam is usually a pretty good optimizer. Feel free to experiment here

              metrics=['accuracy']) # also keep track of the accuracy



# This gives a high-level overview of the model setup

model.summary()
# train the model

from keras.callbacks import EarlyStopping



# This evaluates after every epoch to ensure that we are still improving on our validation set.

# Otherwise, it will stop fitting the model and restore the condition with the best weights.

earlyStopping = EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=2)



history = model.fit(X_train, y_train,

              batch_size=32,

              epochs=50,

              validation_data=(X_val, y_val),

              callbacks = [earlyStopping],

              shuffle=True)
# Plot the performance

plt.figure(figsize=(10,5))

plt.subplot(1,2,1) # many of the commands to plt are like the matlab plotting functions

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.subplot(1,2,2) 

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])
# Score trained model.

y_test_pred = model.predict_classes(X_test, verbose=1)
# Write your label predictions to a CSV file to submit to Kaggle.

with open('predictions.csv','w') as csvfile:

     fieldnames = ['Index','Label']

     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

     writer.writeheader()    

     for index,l in enumerate(y_test_pred):

         writer.writerow({'Index': str(index), 'Label': str(l)})

print("Predictions have been generated.")
model_name = 'mlp_example_for_kaggle'
