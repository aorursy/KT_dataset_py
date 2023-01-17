

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input/"))

import cv2

import matplotlib.pyplot as plt 

import seaborn as sns

import os

from PIL import Image

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.utils import np_utils

from skimage import exposure, color

from skimage.transform import resize

print(os.listdir("../input/cell_images/cell_images/Parasitized"))
parasitized_path = "../input/cell_images/cell_images/Parasitized/"

uninfected_path = "../input/cell_images/cell_images/Uninfected/"
parasitized_cell = os.listdir(parasitized_path)

print(parasitized_cell[:5]) 
uninfected_cell = os.listdir(uninfected_path)

print(uninfected_cell[:5])
plt.figure(figsize = (10,10))

for i in range(6):

    plt.subplot(2, 3, i+1)

    img = cv2.imread(parasitized_path+ parasitized_cell[i])

    plt.imshow(img)

    plt.title('Parasitized')

    plt.axis('off')

plt.show()
plt.figure(figsize = (10,10))

for i in range(6):

    plt.subplot(2, 3, i+1)

    img = cv2.imread(uninfected_path + uninfected_cell[i])

    plt.imshow(img)

    plt.title('Uninfected')

    plt.axis('off')

plt.show()
data = []

labels = []

for image in parasitized_cell:

    try:

        read_image = plt.imread(parasitized_path + image)

        resize_image = cv2.resize(read_image, (50, 50))

        array_image = img_to_array(resize_image)

        data.append(array_image)

        labels.append(1)

    except:

        None

        

for image in uninfected_cell:

    try:

        read_image = plt.imread(uninfected_path + image)

        resize_image = cv2.resize(read_image, (50, 50))

        array_image = img_to_array(resize_image)

        data.append(array_image)

        labels.append(0)

    except:

        None
data
labels
plt.imshow(data[1])

plt.show
plt.imshow(data[20000])

plt.show
image_data = np.array(data)

labels = np.array(labels)
image_data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2)
y_train = np_utils.to_categorical(y_train,  2)

y_test = np_utils.to_categorical(y_test,  2)
print("Shape of training image : ", X_train.shape)

print("Shape of testing image : ", X_test.shape)

print("Shape of training labels : ", y_train.shape)

print("Shape of testing labels : ", y_test.shape)
import keras

from keras.layers import Dense, Conv2D

from keras.layers import Flatten

from keras.layers import MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Dropout

from keras.models import Sequential

from keras import backend as K

from keras import regularizers

from keras import optimizers
#Defining the model

def CNNbuild(height, width, classes, channels):

    model = Sequential()

    

    inputShape = (height, width, channels)

    chanDim = -1

    

    if K.image_data_format() == 'channels_first':

        inputShape = (channels, height, width)

    model.add(Conv2D(40,kernel_size=(3, 3), activation = 'relu', padding="same", input_shape = inputShape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(BatchNormalization(axis = chanDim))





    model.add(Conv2D(70,kernel_size=(3,3), activation = 'relu', padding="same"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(BatchNormalization(axis = chanDim))





    model.add(Conv2D(500,kernel_size=(3,3) , activation = 'relu', padding="same"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(BatchNormalization(axis = chanDim))



    model.add(Flatten())

    

    model.add(Dense(units=100, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(units=200, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(units=300, activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(1024, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02), activation = 'relu' ))

    model.add(BatchNormalization(axis = chanDim))

    model.add(Dropout(0.5))

    model.add(Dense(classes, activation = 'sigmoid'))

    

    return model

#instantiate the model

height = 50

width = 50

classes = 2

channels = 3

model = CNNbuild(height = height, width = width, classes = classes, channels = channels)

model.summary()
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   horizontal_flip = True,

                                   vertical_flip = True,

                                   rotation_range=0.2,

                                   shear_range=0.2,

                                   )



test_datagen = ImageDataGenerator(rescale = 1./255)
train_datagen.fit(X_train)
training_set = train_datagen.flow(X_train,

                                  y_train,

                                 batch_size = 32)



test_set = test_datagen.flow(X_test,

                             y_test,

                             batch_size = 32)
#Compiling the Model

model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy'])
history_epochs = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25)
# summarize history for accuracy

plt.plot(history_epochs.history['acc'])

plt.plot(history_epochs.history['val_acc'])

plt.title('Model Accuracy Comparison')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train','Test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history_epochs.history['loss'])

plt.plot(history_epochs.history['val_loss'])

plt.title('Model Loss Comparison')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train','Test'], loc='upper right')

plt.show()
# make predictions on the test set

preds = model.predict(X_test)
from sklearn.metrics import accuracy_score



print(accuracy_score(y_test.argmax(axis=1), preds.argmax(axis=1)))
from sklearn.metrics import classification_report

print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1)))
from sklearn.metrics import log_loss

logloss = log_loss(y_test.argmax(axis=1), preds)

print(logloss)