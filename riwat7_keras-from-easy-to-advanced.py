#import necessary modules

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow import keras



seed =1 #for reproductability

np.random.seed(seed)



# read data

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(train.shape)

print(test.shape)
# use float values

Xtrain = (train.iloc[:, 1:].values).astype('float32')

ytrain = (train.iloc[:, 0].values).astype('int32')

Xtest = test.values.astype('float32')



#reshape data into 28x28 samples

Xtrain = Xtrain.reshape(Xtrain.shape[0],28,28)

Xtest = Xtest.reshape(Xtest.shape[0],28,28)

print(Xtrain.shape)

print(Xtest.shape)
# example of images

def showImages(X, n):

  images = X.shape[0]

  steps = n*n

  step = int(images/steps)



  plt.figure(figsize=(7,7))

  plt.title('Examples of images')



  for i in range(steps):

    plt.subplot(n,n,i+1)

    plt.imshow(X[i*step])

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

  plt.show()



showImages(Xtrain, 12)
#standarize

mean = Xtrain.mean().astype(np.float32)

std = Xtrain.std().astype(np.float32)

def standarize(X):

  return (X-mean)/std



Xtrain = standarize(Xtrain)

Xtest = standarize(Xtest)

#build categorical output

print("output exaples:", ytrain[0:20])

ytrain = keras.utils.to_categorical(ytrain)

num_classes = ytrain.shape[1]

print("output categorized examples", ytrain[0:5])
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.10, random_state=1)

print(Xtrain.shape)

print(Xval.shape)
#simple multiperceptron model

def MultiPerceptronModel():

  model = keras.Sequential()

  model.add(keras.layers.Flatten(input_shape=(28,28)))

  model.add(keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer=keras.optimizers.Adam(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])

  return model

model=MultiPerceptronModel()

print(model.summary())
#set number of epochs

ep = 5    

hist = model.fit(Xtrain, ytrain, epochs=ep, verbose=1)

print("loss:", hist.history['loss'][ep-1], "acc:", hist.history['accuracy'][ep-1])
def validate(model, Xval):

  ypred = model.predict_classes(Xval, verbose=0)

  metrics = keras.metrics.Accuracy()

  metrics.update_state(ypred, np.argmax(yval, axis=1))

  print("Validation accuracy:", metrics.result().numpy())



validate(model, Xval)
def NeuralNetworkModel():

  model = keras.Sequential()

  model.add(keras.layers.Flatten(input_shape=(28,28)))

  model.add(keras.layers.Dense(512, activation='relu'))

  model.add(keras.layers.Dense(256, activation='relu'))

  model.add(keras.layers.Dense(96, activation='relu'))

  model.add(keras.layers.Dense(24, activation='relu'))

  model.add(keras.layers.Dense(10, activation='softmax'))

  print("input shape ",model.input_shape)

  print("output shape ",model.output_shape)

  #model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),

  model.compile(optimizer=keras.optimizers.Adam(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])

  return model;



model=NeuralNetworkModel()

print(model.summary())
#fit model

ep = 5    

hist = model.fit(Xtrain, ytrain, epochs=ep, verbose=1)

print("loss:", hist.history['loss'][ep-1], "acc:", hist.history['accuracy'][ep-1])

#validate model

validate(model, Xval)
def ConvNeuralNetworkModel():

  model = keras.Sequential()

  model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))

  model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))

  model.add(keras.layers.MaxPooling2D())

  model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))

  model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))

  model.add(keras.layers.MaxPooling2D())

  model.add(keras.layers.Flatten())

  model.add(keras.layers.Dense(512, activation='relu'))

  model.add(keras.layers.Dense(10, activation='softmax'))

  #gen = keras.preprocessing.image.ImageDataGenerator()

  model.compile(optimizer=keras.optimizers.Adam(lr=0.001),

                loss='categorical_crossentropy',

                metrics=['accuracy'])

  return model



model=ConvNeuralNetworkModel()
Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28,1)

Xval = Xval.reshape(Xval.shape[0], 28, 28,1)

Xtest = Xtest.reshape(Xtest.shape[0], 28, 28,1)

print(Xtrain.shape)
ep = 5   

#verbose is set to verbose=0, because it generates to much logs

hist = model.fit(Xtrain, ytrain, epochs=ep, verbose=0)

print("loss:", hist.history['loss'][ep-1], "acc:", hist.history['accuracy'][ep-1])

#validate model

validate(model, Xval)

print(model.summary())
def ConvNeuralNetworkFitModel():

  fitlayer = keras.layers.Dropout(0.1)

  #fitlayer = keras.layers.BatchNormalization()

  model = keras.Sequential()

  model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))

  model.add(fitlayer)

  model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))

  model.add(keras.layers.MaxPooling2D())

  model.add(fitlayer)

  model.add(keras.layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))

  model.add(fitlayer)

  model.add(keras.layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))

  model.add(keras.layers.MaxPooling2D())

  model.add(fitlayer)

  model.add(keras.layers.Flatten())

  model.add(keras.layers.Dense(512, activation='relu'))

  model.add(fitlayer)

  model.add(keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer=keras.optimizers.Adam(lr=0.001),

                loss='categorical_crossentropy',

                metrics=['accuracy'])

  return model;



model = ConvNeuralNetworkFitModel()
#use image generator

gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, shear_range=0.3,

                         height_shift_range=0.1, zoom_range=0.1)

batches = gen.flow(Xtrain, ytrain, batch_size=64)

val_batches = gen.flow(Xval, yval, batch_size=64)

#set number of epochs

ep = 5  

#verbose is set to verbose=0, because it generates to much logs

model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=ep, verbose=0,

                    validation_data=val_batches, validation_steps=val_batches.n)

print("loss:", hist.history['loss'][ep-1], "acc:", hist.history['accuracy'][ep-1])

#validate model

validate(model, Xval)