import json, sys, random

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Flatten, Activation

from keras.layers import Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.optimizers import SGD

import keras.callbacks

from matplotlib import pyplot as plt

from PIL import Image
f = open(r'../input/ships-in-satellite-imagery/shipsnet.json')

dataset = json.load(f)

f.close()
input_data = np.array(dataset['data']).astype('uint8')

output_data = np.array(dataset['labels']).astype('uint8')

input_data.shape
n_spectrum = 3

weight = 80

height = 80

X = input_data.reshape([-1, n_spectrum, weight, height])

X[0].shape
label={0:'no hay barco',1:'obvi bobis'}

plt.subplot(131)

plt.title(label[int(output_data[0])])

plt.imshow(Image.fromarray( ((np.array(input_data[0]).astype('uint8')).reshape((3, 6400)).T.reshape((80,80,3)))));

plt.subplot(132)

plt.title(label[int(output_data[1])])

plt.imshow(Image.fromarray( ((np.array(input_data[1]).astype('uint8')).reshape((3, 6400)).T.reshape((80,80,3)))));

plt.subplot(133)

plt.title(label[int(output_data[2])])

plt.imshow(Image.fromarray( ((np.array(input_data[2]).astype('uint8')).reshape((3, 6400)).T.reshape((80,80,3)))));
output_data.shape
output_data
y = np_utils.to_categorical(output_data, 2)
indexes = np.arange(2800)

np.random.shuffle(indexes)
X_train = X[indexes].transpose([0,2,3,1])

y_train = y[indexes]
X_train = X_train / 255
np.random.seed(42)
model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25))



model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25))



model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25))



model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



model.fit(X_train, y_train,

    batch_size=32,

    epochs=18,

    validation_split=0.2,

    shuffle=True,

    verbose=2)
model.save('barquito_modelo.h5py')

model.save_weights('barquito_peso.h5py')
from keras.models import load_model 

modelo=load_model('barquito_modelo.h5py')
from keras.preprocessing import image

im=image.load_img(r'../input/ships-in-satellite-imagery/shipsnet/shipsnet/1__20180713_180403_1035__-118.22515217787108_33.73850035178332.png',target_size=(80,80))

im
imagen=image.img_to_array(im)

imagen=imagen.reshape(-1,80,80,3)

s=model.predict_classes(imagen)

print('La imagen se muestra que',label[int(s)])