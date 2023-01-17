import struct

import random

import numpy as np 

import pandas as pd

import seaborn as sns

import tensorflow as tf

from array import array

from os.path  import join

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
X_train = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")

X_test = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")



#Extracting the labels from the data

y_train = X_train["label"]

y_test = X_test["label"]



#Dropping the labels from the X_train and X_test dataframe

X_train.drop(columns = ["label"], inplace = True)

X_test.drop(columns = ["label"], inplace = True)



#Checking the shapes of the dataframes

X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train = X_train/255

X_test = X_test/255
#Reshaping the data from (m,784) to (m,28,28)

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
plt.figure(figsize = (2,2))

plt.imshow(X_train[108].reshape(28, 28) , cmap = "gray");
#Augmenting the image data to prevent overfitting

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
plt.figure(figsize = (12,5))

sns.countplot(y_train);
def getCNNModel(shape):

    

    model = Sequential([

        Conv2D(64, (3,3), strides = 1, padding = "same", input_shape = shape, activation = "relu"),

        BatchNormalization(),

        MaxPool2D((2,2), strides =2, padding = "same"),

        Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),

        Dropout(0.2),

        BatchNormalization(),

        MaxPool2D((2,2) , strides = 2 , padding = 'same'),

        Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),

        BatchNormalization(),

        MaxPool2D((2,2) , strides = 2 , padding = 'same'),

        Flatten(),

        Dense(512 , activation = 'relu'),

        Dropout(0.3),

        Dense(units = 25 , activation = 'softmax')

    ])

    

    return model
model = getCNNModel(X_train[0].shape)

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
history = model.fit(datagen.flow(X_train,y_train, batch_size = 128) ,epochs = 20 , validation_data = (X_val, y_val) , 

                    callbacks = [learning_rate_reduction])
print("Test Accuracy " , model.evaluate(X_test,y_test)[1]*100 , "%")
y_pred = model.predict_classes(X_test)
dfConfMat = pd.DataFrame(confusion_matrix(y_test,y_pred))

plt.figure(figsize = (15,15))

sns.heatmap(dfConfMat,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='');