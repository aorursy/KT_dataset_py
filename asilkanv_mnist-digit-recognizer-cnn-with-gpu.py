import os

import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import time

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 

# put labels into y_train variable

Y_train = train["label"]

# free some space

del train 

X_train.head()
Y_train.nunique() # We have 10 classes which are in the range of 0-9 numbers
# normalization with 255 because max value in image is 255



X_train = X_train/255.0

test = test / 255.0
#Reshape

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
plt.imshow(X_train[0][:,:,0], cmap = "gray")

plt.show()
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state= 0)
dense_layers = [1]

layer_sizes = [64]

conv_layers = [3]



for dense_layer in dense_layers: 

    for layer_size in layer_sizes:

         for conv_layer in conv_layers:

            model =  Sequential()



            model.add(Conv2D(layer_size, (3,3), input_shape = (28,28,1), padding = 'Same'))

            model.add(Activation("relu"))

            model.add(MaxPooling2D(pool_size = (2,2)))



            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3,3), padding = 'Same'))

                model.add(Activation("relu"))

                model.add(MaxPooling2D(pool_size = (2,2)))

                model.add(Dropout(0.25))

                

            model.add(Flatten()) # converts ourr 3D feature maps to 1D feature vectors



            for l in range(dense_layer):

                model.add(Dense(layer_size))

                model.add(Activation("relu"))



            model.add(Dense(10)) # output layer

            model.add(Activation("softmax"))

model.summary()
model.compile(optimizer = "adam" , loss = "categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(

        rotation_range=8,  # randomly rotate images

        zoom_range = 0.5, # Randomly zoom image 

        width_shift_range=0.08,  # randomly shift images horizontally 

        height_shift_range=0.08,  # randomly shift images vertically 

        shear_range = 0.5,

        ) 



datagen.fit(X_train)



# fitting model

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= 32),

                              epochs = 10, validation_data = (X_val, Y_val), steps_per_epoch= X_train.shape[0] // 32, validation_steps = X_val.shape[0] // 32)
y_pred = model.predict(X_val)
for i in range(5):

    plt.grid(False)

    plt.imshow(X_val[i][:,:,0], cmap = plt.cm.binary)

    plt.xlabel("Actual: " + str(np.argmax(Y_val[i])))

    plt.title("Prediction" + str(np.argmax(y_pred[i])))

    plt.show()
# predict results

results = model.predict(test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)