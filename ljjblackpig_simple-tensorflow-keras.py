from __future__ import absolute_import, division, print_function



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns



print(tf.__version__)
#Read in the dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(3)
test.head(3)
#Visualizing the label column

sns.countplot(train['label'])
#Grab the response variable label column

label = train.iloc[:, train.columns == 'label']

train = train.iloc[:, train.columns != 'label']
label.info()
#Convert dataframe object to an ndarray and reshape them

train_image = train.values.reshape(-1, 28, 28, 1)

test_image = test.values.reshape(-1, 28, 28, 1)



print(train_image.shape, test_image.shape)
#Now we have structured our data, we can check the imaging

plt.imshow(train_image[1][:,:,0])
#Now we scale our data to fasten the training process

train_images = train_image / 255.0



test_images = test_image / 255.0
#Visualizing plots again

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i][:,:,0], cmap=plt.cm.binary)

plt.show()
#Prepare the data for model building

#Separate into training and validation set

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(

               train_images, label.values, test_size = 0.25, random_state = 2019)
model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'Same',  activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'Same',  activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = 'relu'))



model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer='adam',  #Autoadjust optimizer

              loss='sparse_categorical_crossentropy', #Cross entropy function for loss function

              metrics=['accuracy'])
history = model.fit(X_train, y_train, 

          validation_data = (X_val, y_val),

          verbose = 2,

          epochs=25)
#Graphing the training loss / validation loss to check overfitting or underfitting

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
#A visulization of our Neural Network

from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image("model.png")
#Predict the model

results = model.predict(test_images)