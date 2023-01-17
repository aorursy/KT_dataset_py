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



import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline



from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape

from keras.models import Sequential

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from keras.layers import Dense , Activation

from keras.layers import Dropout

from keras.layers import Flatten

from keras.constraints import maxnorm

from keras.optimizers import SGD , Adam

from keras.layers import Conv2D , BatchNormalization

from keras.layers import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

K.set_image_dim_ordering('th')

x = pd.read_csv('../input/fer2013.csv')

print (x.values.shape)

x.head(5)
data = x.values

print (data.shape)



y = data[:, 0]

pixels = data[:, 1]

pixels.shape
X = np.zeros((pixels.shape[0], 48*48))



for ix in range(X.shape[0]):

    p = pixels[ix].split(' ')

    for iy in range(X.shape[1]):

        X[ix, iy] = int(p[iy])

X = X/255

print(X.shape)

print(y.shape)

# for ix in range(4):

#     plt.figure(ix)

#     plt.imshow(X[ix].reshape((48, 48)), interpolation='none', cmap='gray')

# plt.show()
X_train = X[0:28710, :]

Y_train = y[0:28710]

print (X_train.shape, Y_train.shape)

X_crossval = X[28710:32300, :]

Y_crossval = y[28710:32300]

print (X_crossval.shape, Y_crossval.shape)

X_train = X_train.reshape((X_train.shape[0], 1 , 48, 48 ))

X_crossval = X_crossval.reshape((X_crossval.shape[0], 1 ,48, 48))

X_train.shape

print (y.shape)

y_ = np_utils.to_categorical(y, 7)

print (y_.shape)



Y_train = y_[:28710]

Y_crossval = y_[28710:32300]

print (X_crossval.shape, Y_crossval.shape)

datagen = ImageDataGenerator(

        featurewise_center=True,  

        samplewise_center=False,  

        featurewise_std_normalization=True,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=20,  

        zoom_range = 0.0,  

        width_shift_range=0.2,  

        height_shift_range=0.2,  

        horizontal_flip=True, 

        vertical_flip=False)  



datagen.fit(X_train)
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=1, verbose=1)
def swish_activation(x):

    return (K.sigmoid(x) * x)



model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(1,48,48)))

model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))

model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))

# model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# model.add(Dense(512, activation=swish_activation))

# model.add(Dropout(0.4))

model.add(Dense(64, activation=swish_activation))

model.add(Dropout(0.4))

model.add(Dense(7 , activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam' ,

              metrics=['accuracy'])



print(model.summary())
batch_size = 128

epochs = 14



model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

steps_per_epoch = len(X) // batch_size

validation_steps = len((X_crossval, Y_crossval)) // batch_size
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                    steps_per_epoch=X.shape[0] // batch_size,

                    callbacks=[lr_reduce],

                    validation_data=(X_crossval, Y_crossval),

                    epochs = epochs, verbose = 2)



weights = model.get_weights()

np.save('my_model_weights', weights)
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