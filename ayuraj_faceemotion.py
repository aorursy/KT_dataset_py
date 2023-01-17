import numpy as np 

import pandas as pd

import os

import cv2

import random

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from sklearn.model_selection import train_test_split



import keras.backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense, Flatten, BatchNormalization, Dropout

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint
dataPath = '../input/fer2013.csv'
data = pd.read_csv(dataPath)
data.head(10)
emotions = data['emotion'].values
set(emotions)
plt.figure(figsize=(13,7))

plt.hist(emotions, bins=7)

plt.grid(True)

plt.show()
def groupY(y):

    if y==0:

        return 0

    elif y==3:

        return 1

    elif y==4:

        return 2

    else:

        return 3

    

def decodeY(y):

    if y==0:

        return 'Angry'

    elif y==1:

        return 'Happy'

    elif y==2:

        return 'Sad'

    else:

        return 'Neutral'
def createData(data, test_size):

    data = data.values

    y = data[:, 0]

    pixels = data[:, 1]

    

    data_discarded = 0

    X = []

    Y = []

    count = 0

    for ix in range(pixels.shape[0]):

        if y[ix]==0 or y[ix]==3 or y[ix]==4 or y[ix]==6:

            if count%1000 == 0:

                print("[INFO] {} images loaded".format(count))

            temp = np.zeros((48*48))

            p = pixels[ix].split(' ')

            for iy in range(temp.shape[0]):

                temp[iy] = int(p[iy])

            X.append(temp)

            Y.append(groupY(y[ix]))

            count+=1

        else: 

            data_discarded+=1



    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=True)  

    y_train = to_categorical(y_train)

    y_test = to_categorical(y_test)

    print("[INFO] Done")

    

    return np.array(X_train), np.array(X_test), y_train, y_test
X_train, X_test, y_train, y_test = createData(data, 0.2)
def showImage(X, y):

    for ix in range(4):

        plt.figure(ix)

        plt.title(decodeY(np.argmax(y[ix])))

        plt.imshow(X[ix].reshape((48, 48)), interpolation='none', cmap='gray')

    plt.show()
showImage(X_train, y_train)
X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))

X_test = X_test.reshape((X_test.shape[0], 48, 48, 1))
print("Training Data")

print("X_train shape: ", X_train.shape)

print("y_train shape: ", y_train.shape)

print("Test Data")

print("X_test shape: ", X_test.shape)

print("y_test shape: ", y_test.shape)
K.clear_session()
def createModel():



    inputs = Input(shape=(48,48,1))

    x = Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)

    x = Conv2D(32, (3, 3), padding="same", activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.5)(x)

                     

    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)

    x = Conv2D(64, (3, 3), padding="valid", activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.5)(x)

    

    x = Conv2D(96, (3, 3), padding="same", activation='relu')(x)

    x = Conv2D(96, (3, 3), padding="valid", activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.5)(x)

    

    x = Conv2D(128, (3, 3), dilation_rate=(2, 2), padding="same", activation='relu')(x)

    x = Conv2D(128, (3, 3), padding="valid", activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.5)(x)

    

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)

#     x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)

#     x = Dropout(0.4)(x)

    x = Dense(4 , activation='softmax')(x)

    

    model = Model(inputs=inputs, outputs=x)

    

    return model
model = createModel()
model.summary()
batch_size = 64

epochs = 100
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

steps_per_epoch = len(X_train) // batch_size

validation_steps = len(X_test) // batch_size
datagen = ImageDataGenerator()
earlyStopper = EarlyStopping(monitor='val_loss', patience=10)
reduceLRrate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                    steps_per_epoch=steps_per_epoch,

                    validation_data=(X_test, y_test),

                    validation_steps=validation_steps,

                    epochs = epochs,

                    callbacks=[earlyStopper])
model.save('emotion.hdf5')
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()