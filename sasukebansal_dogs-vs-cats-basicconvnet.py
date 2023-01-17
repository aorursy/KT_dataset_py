# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2                 

import os                  

from tqdm import tqdm, tqdm_notebook

from random import shuffle



train_directory= '../input/dogs-vs-cats/train/train'

test_directory= '../input/dogs-vs-cats/test1/test1'
image_height= 64

image_width= 64
import glob

train_cats = sorted(glob.glob(os.path.join(train_directory, 'cat*.jpg')))

train_dogs = sorted(glob.glob(os.path.join(train_directory, 'dog*.jpg')))
len(train_cats), len(train_dogs)
shuffle(train_cats)

shuffle(train_dogs)

#Creating training and validation splits

train_paths = train_cats[:10000] + train_dogs[:10000]

val_paths = train_cats[10000:] + train_dogs[10000:]
len(train_paths), len(val_paths)
#Function for defining label

import re



def label_img(path):

    if re.search(r"cat\.", path):

        return np.array([0, 1])

    elif re.search(r"dog\.", path):

        return np.array([1, 0])
#Function for loading and resizing the images

def create_image_dataset(path_list, image_height, image_width, grayscale=True):

    images=[]

    labels=[]

    

    for path in tqdm_notebook(path_list):

        label = label_img(path)

        

        if grayscale:

            image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        else:

            image = cv2.imread(path)

            

        image = cv2.resize(image, (image_height,image_width))

        

        images.append(np.array(image))

        labels.append(label)

        

    return np.array(images), np.array(labels)
train_paths[6]
#Creating array with data

X, y = create_image_dataset(train_paths, image_height, image_width)

X_val, y_val = create_image_dataset(val_paths, image_height, image_width)
print(X.shape)

print(y.shape)



print(X_val.shape)

print(y_val.shape)
y_val.sum(axis=0)
#Creating training ad Validation arrays

X = X.reshape(-1,image_height,image_width, 1)

X_val = X_val.reshape(-1,image_height,image_width, 1)

#Scaling data for neural network

X = X/float(255)

X_val = X_val/float(255)
X.shape, X_val.shape
#Importing keras libraries

import keras

from keras.layers import Input, Dense, Flatten, Dropout

from keras.layers import MaxPooling2D, Conv2D

from keras.optimizers import SGD, Adam

from keras.models import Model
keras.backend.clear_session()

#Network Architecture

x_input = Input(shape=(image_height, image_width, 1))

#x = Flatten()(x_input)

#x = Dense(4096, activation="relu")(x)

#x = Dropout(0.2)(x)

#x = Dense(8096, activation="relu")(x)

x = Conv2D(32, (3, 3), padding="same", activation='relu')(x_input)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



#Additional Layer

x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)

# x = MaxPooling2D(pool_size=(2, 2))(x)



# x = Conv2D(32, (3, 3), padding="same", activation='relu')(x)

# x = MaxPooling2D(pool_size=(2, 2))(x)



x = Flatten()(x)

#Additional Layer

x = Dense(256, activation="relu")(x)



#x = Dropout(0.3)(x)



x = Dense(16, activation="relu")(x)

x_out = Dense(2, activation="softmax")(x)



#Specifying input and output

model = Model(inputs=x_input, outputs=x_out)
sgd = SGD(lr=0.01, momentum=0.7, decay=0.00, nesterov=False)

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



model.compile(optimizer=sgd, loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()
history= model.fit(x=X, 

                    y=y, 

                    validation_data=(X_val, y_val), 

                    epochs = 100, 

                    batch_size=128, 

                    verbose=1)
import matplotlib.pyplot as plt



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()