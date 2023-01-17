from tensorflow import keras as keras

import tensorflow as tf

import csv

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline
IMAGE_WIDTH = 96

IMAGE_HEIGHT = 96
def load_dataset():

    '''

    Load training dataset

    '''

    Xtrain = []

    Ytrain = []

    with open('../input/training/training.csv') as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:

            img = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,1), dtype=np.float)

            for i, val in enumerate(row["Image"].split(" ")):

                img[i//IMAGE_WIDTH,i%IMAGE_WIDTH,0] = val

            Yitem = []

            failed = False

            for coord in row:

                if coord == "Image":

                    continue

                if(row[coord].strip()==""):

                    failed = True

                    break

                Yitem.append(float(row[coord]))

            if not failed:

                Xtrain.append(img)

                Ytrain.append(Yitem)

                

    return np.array(Xtrain), np.array(Ytrain, dtype=np.float)
# Load dataset

Xdata, Ydata = load_dataset()

Xtrain = Xdata[:]

Ytrain = Ydata[:]
def show_image(X, Y):

    img = np.copy(X)

    for i in range(0,Y.shape[0],2):

        if 0 < Y[i+1] < IMAGE_HEIGHT and 0 < Y[i] < IMAGE_WIDTH:

            img[int(Y[i+1]),int(Y[i]),0] = 255

    plt.imshow(img[:,:,0])
# Preview dataset samples

show_image(Xtrain[0], Ytrain[0])
# Configure Model

from keras.models import Sequential

from keras.layers import Conv2D, Convolution2D,Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D

from keras.layers import Flatten, Dense

from keras.layers.normalization import BatchNormalization





# Build a CNN architecture



model = Sequential()

model.add(BatchNormalization(input_shape=(96, 96, 1)))

model.add(Convolution2D(24, 5, 5, border_mode="same", init="he_normal", input_shape=(96, 96, 1), dim_ordering="tf"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

model.add(Convolution2D(36, 5, 5))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

model.add(Convolution2D(48, 5, 5))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

model.add(Convolution2D(64, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

model.add(Convolution2D(64, 3, 3))

model.add(Activation('relu'))

model.add(GlobalAveragePooling2D());

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(30))

# Summarize the model

model.summary()
from keras.callbacks import ModelCheckpoint, History

from keras.optimizers import Adam



hist = History()

epochs = 1100

batch_size = 64





## TODO: Compile the model

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])



hist_final = model.fit(Xtrain, Ytrain, validation_split=0.2,

          epochs=epochs, batch_size=batch_size, callbacks=[hist], verbose=1)

plt.plot(range(epochs), hist_final.history[

         'val_loss'], 'g-', label='Val Loss')

plt.plot(range(epochs), hist_final.history[

         'loss'], 'g--', label='Train Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Load test data

def load_testset():

    Xtest = []

    with open('../input/test/test.csv') as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:

            img = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,1), dtype=np.float)

            for i, val in enumerate(row["Image"].split(" ")):

                img[i//IMAGE_WIDTH,i%IMAGE_WIDTH,0] = val

            Xtest.append(img)

                

    return np.array(Xtest)

Xtest = load_testset()
# Preview results on test data

def show_results(image_index):

    Ypred = model.predict(Xtest[image_index:(image_index+1)])

    show_image(Xtest[image_index], Ypred[0])
show_results(3)
show_results(4)
show_results(5)
model.save("points.h5")