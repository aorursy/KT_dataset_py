import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

import random

import pickle

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Conv2D,Activation,Flatten,MaxPooling2D

from sklearn.model_selection import train_test_split

import gc
DATADIR = '/kaggle/input/cats-and-dogs-sentdex-tutorial/PetImages/'

CATEGORIES = ['Cat','Dog'] #CAT=0, DOG=1

IMG_SIZE = 100

# os.listdir('/kaggle/working')
training_data = []

def create_training_data():

    for c in CATEGORIES:

        for img in os.listdir(os.path.join(DATADIR,c)):

            try:

                img_array = cv2.imread(os.path.join(os.path.join(DATADIR,c),img))

                img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

                training_data.append([img_array,CATEGORIES.index(c)])

                break

                #plt.imshow(img_array,cmap='gray')

            except Exception as e:

                pass

            

create_training_data()
random.shuffle(training_data)
def getxy():

    Xx = [_[0] for _ in training_data]

    Yy = [_[1] for _ in training_data]

    X = np.array(Xx).reshape(-1,IMG_SIZE,IMG_SIZE,3) /255

    Y = np.array(Yy)

    return X,Y
X,Y = getxy()

del training_data

gc.collect()
# handlex = open('/kaggle/working/x.pickle','wb')

# handley = open('/kaggle/working/y.pickle','wb')

# pickle.dump(X,handlex)

# pickle.dump(Y,handley)

# handlex.close()

# handley.close()
# handlex = open('/kaggle/working/x.pickle','rb')

# handley = open('/kaggle/working/y.pickle','rb')

# X = pickle.load(handlex)

# Y = pickle.load(handley)

# handlex.close()

# handley.close()

def getmodel():

    model = Sequential()



    model.add(Conv2D(256,(3,3),input_shape=X.shape[1:]))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Conv2D(256,(2,2)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(128,(2,2)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(128,(2,2)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))





    model.add(Flatten())

    model.add(Dense(128))

    model.add(Activation('relu'))





    model.add(Dense(64))

    model.add(Activation('relu'))



    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
model = getmodel()
model.fit(X,Y,batch_size=64, validation_split=0.1, epochs=0)
del model
gc.collect()