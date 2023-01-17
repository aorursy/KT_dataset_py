# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2                 
import os                  
from tqdm import tqdm
from random import shuffle
# Unzipping the input folder using command. 
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train
!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
TRAIN_DIR = '../working/train/train/'
TEST_DIR = '../working/test/test/'
IMG_SIZE = 64
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR + i for i in os.listdir(TEST_DIR)]
len(train_cats), len(train_dogs), len(train_images)
# each record is path and name of the image. 
train_cats[0]
# Creating training and validation splits
train_list = train_cats[:10000] + train_dogs[:10000]
val_list = train_cats[10000:] + train_dogs[10000:]
len(train_list), len(val_list)
#Function for defining label
def label_img(img):
    if 'cat' in img: return [0, 1]
    elif 'dog' in img: return [1, 0]
#Function for resizing images
def create_train_data(train_list):
    training_data = []
    for img in tqdm(train_list):
        label = label_img(img)
        path = img
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),label])
    shuffle(training_data)
    return training_data
train_list[0]
#Creating array with data
train = create_train_data(train_list)
val = create_train_data(val_list)
train[1000][1]
#Creating training ad Validation arrays
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE)
Y = [i[1] for i in train]

val_X = np.array([i[0] for i in val]).reshape(-1,IMG_SIZE,IMG_SIZE)
val_Y = [i[1] for i in val]
train[0]
import matplotlib.pyplot as plt
print('Label is ', train[105][1])
plt.imshow(train[105][0])
#Scaling data for neural network
X = X/float(255)
val_X = val_X/float(255)
X.shape
Y = np.asarray(Y)
val_Y = np.asarray(val_Y)
X.shape, val_X.shape
Y.shape, val_Y.shape
#Importing keras libraries
import keras
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.models import Model
keras.backend.clear_session()

# Network Architecture
x_input = Input(shape=(IMG_SIZE, IMG_SIZE))
x = Flatten()(x_input)
x = Dense(1024, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x_out = Dense(2, activation="softmax")(x)

#Specifying input and output
model = Model(inputs=x_input, outputs=x_out)
# SGD with no momentum
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(x=X, 
                    y=Y, 
                    validation_data=(val_X, val_Y), 
                    epochs = 30, 
                    batch_size=128, 
                    verbose=1)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();
# Running for more number of epochs
keras.backend.clear_session()

# Network Architecture
x_input = Input(shape=(IMG_SIZE, IMG_SIZE))
x = Flatten()(x_input)
x = Dense(1024, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x_out = Dense(2, activation="softmax")(x)

#Specifying input and output
model = Model(inputs=x_input, outputs=x_out)
# SGD with no momentum
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss = "binary_crossentropy", metrics=["accuracy"])
history = model.fit(x=X, 
                    y=Y, 
                    validation_data=(val_X, val_Y), 
                    epochs = 80, 
                    batch_size=128, 
                    verbose=1)
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();
# Using momentum (usually helps with faster convergence)
keras.backend.clear_session()

# Network Architecture
x_input = Input(shape=(IMG_SIZE, IMG_SIZE))
x = Flatten()(x_input)
x = Dense(1024, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x_out = Dense(2, activation="softmax")(x)

#Specifying input and output
model = Model(inputs=x_input, outputs=x_out)
# SGD with no momentum
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss = "binary_crossentropy", metrics=["accuracy"])
history = model.fit(x=X, 
                    y=Y, 
                    validation_data=(val_X, val_Y), 
                    epochs = 80, 
                    batch_size=128, 
                    verbose=1)
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();
# Using Adam Optimizer (Adam can converge faster but sometimes may have convergence issues.)
# SGD + momentum can have better convergence than Adam (but Adam is more fast)
keras.backend.clear_session()

# Network Architecture
x_input = Input(shape=(IMG_SIZE, IMG_SIZE))
x = Flatten()(x_input)
x = Dense(1024, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x_out = Dense(2, activation="softmax")(x)

#Specifying input and output
model = Model(inputs=x_input, outputs=x_out)
# SGD with no momentum
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss = "binary_crossentropy", metrics=["accuracy"])
history = model.fit(x=X, 
                    y=Y, 
                    validation_data=(val_X, val_Y), 
                    epochs = 80, 
                    batch_size=512, 
                    verbose=1)
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

# Implementation using simple convolutional network.
import keras
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D, Conv2D
from keras.optimizers import SGD, Adam
from keras.models import Model
keras.backend.clear_session()
#Network Architecture
x_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

x = Conv2D(64, (3, 3), padding="same", activation='relu')(x_input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

#Additional Layer
x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

#Additional Layer
x = Dense(512, activation="relu")(x)

#x = Dropout(0.3)(x)

x = Dense(16, activation="relu")(x)
x_out = Dense(2, activation="softmax")(x)

#Specifying input and output
model = Model(inputs=x_input, outputs=x_out)
sgd = SGD(lr=0.01, momentum=0.8, decay=0.00, nesterov=False)

model.compile(optimizer=sgd, loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(x=X.reshape(X.shape[0], X.shape[1], X.shape[2], 1), 
                    y=Y, 
                    validation_data=(val_X.reshape(val_X.shape[0], val_X.shape[1], val_X.shape[2], 1), val_Y), 
                    epochs = 80, 
                    batch_size=512, 
                    verbose=1)
import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

model.predict(val_X.reshape(val_X.shape[0], val_X.shape[1], val_X.shape[2], 1))
model.predict(val_X.reshape(val_X.shape[0], val_X.shape[1], val_X.shape[2], 1)).argmax(axis= 1)
from keras.applications import VGG16
lower_layers = VGG16(weights= 'imagenet',
                    include_top= False)
keras.backend.clear_session()
#Network Architecture
x_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = lower_layers(x_input)
x = Flatten()(x)

#Additional Layer
x = Dense(256, activation="relu")(x)
x_out = Dense(2, activation="softmax")(x)

#Specifying input and output
model = Model(inputs=x_input, outputs=x_out)
model.summary()
lower_layers.trainable = True
for layer in lower_layers.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    else:
        layer.trainable = False
sgd = SGD(lr=0.01, momentum=0.8, decay=0.00, nesterov=False)

model.compile(loss= 'binary_crossentropy', optimizer= sgd, metrics= ['accuracy'])
#Function for resizing images
def create_train_data(train_list):
    training_data = []
    for img in tqdm(train_list):
        label = label_img(img)
        path = img
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),label])
    shuffle(training_data)
    return training_data


#Creating array with data
train_transfer = create_train_data(train_list)
val_transfer = create_train_data(val_list)

#Creating training ad Validation arrays
X_transfer = np.array([i[0] for i in train_transfer]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y_transfer = [i[1] for i in train_transfer]

val_X_transfer = np.array([i[0] for i in val_transfer]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
val_Y_transfer = [i[1] for i in val_transfer]
#Scaling data for neural network
X_transfer = X_transfer/float(255)
val_X_transfer = val_X_transfer/float(255)

Y_transfer = np.asarray(Y_transfer)
val_Y_transfer = np.asarray(val_Y_transfer)

val_Y_transfer.shape
history = model.fit(x=X_transfer, 
                    y=Y_transfer, 
                    validation_data=(val_X_transfer, val_Y_transfer), 
                    epochs = 50, 
                    batch_size=512, 
                    verbose=1)
import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();
