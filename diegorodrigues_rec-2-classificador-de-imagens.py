import sklearn as skl

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt
basePura = np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_pure.npy")

baseRuido = np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_noisy.npy")

baseGira = np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_rotated.npy")

baseBoth = np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_both.npy")

testImage = np.load("../input/atividade-4-versao-2-fashion-mnist/Test_images.npy")



train_label= pd.read_csv("../input/atividade-4-versao-2-fashion-mnist/train_labels.csv").drop(columns="Id")

train_label.shape
plt.figure(figsize=(15, 7))



plt.subplot(141)

plt.imshow(basePura[0])

plt.subplot(142)

plt.imshow(basePura[1])

plt.subplot(143)

plt.imshow(basePura[2])

plt.subplot(144)

plt.imshow(basePura[3])

plt.show()
plt.figure(figsize=(15, 7))



plt.subplot(141)

plt.imshow(baseGira[0])

plt.subplot(142)

plt.imshow(baseGira[1])

plt.subplot(143)

plt.imshow(baseGira[2])

plt.subplot(144)

plt.imshow(baseGira[3])

plt.show()
plt.figure(figsize=(15, 7))



plt.subplot(141)

plt.imshow(baseRuido[0])

plt.subplot(142)

plt.imshow(baseRuido[1])

plt.subplot(143)

plt.imshow(baseRuido[2])

plt.subplot(144)

plt.imshow(baseRuido[3])

plt.show()
plt.figure(figsize=(15, 7))



plt.subplot(141)

plt.imshow(baseBoth[0])

plt.subplot(142)

plt.imshow(baseBoth[1])

plt.subplot(143)

plt.imshow(baseBoth[2])

plt.subplot(144)

plt.imshow(baseBoth[3])

plt.show()


from __future__ import print_function

import numpy as np

np.random.seed(42)  



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

batch_size = 128

nb_classes = 10

nb_epoch = 10



# dimensões da imagem

img_rows, img_cols = 28, 28

# número de filtros de convolução 

nb_filters = 32



# tamanho do kernel de convolução 

nb_conv = 3
Xpura = basePura.reshape(basePura.shape[0], img_rows, img_cols, 1).astype('float32')

Xruido = baseRuido.reshape(baseRuido.shape[0], img_rows, img_cols, 1).astype('float32')

Xgira = baseGira.reshape(baseGira.shape[0], img_rows, img_cols, 1).astype('float32')

Xboth = baseBoth.reshape(baseBoth.shape[0], img_rows, img_cols, 1).astype('float32')



Xtest = testImage.reshape(testImage.shape[0], img_rows, img_cols, 1).astype('float32')



labelCat = np_utils.to_categorical(train_label)





Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xpura,labelCat, test_size = 0.25)



model = Sequential()



model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid',input_shape=(img_rows, img_cols, 1)))

model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(Xtrain, Ytrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(Xtest, Ytest))



score = model.evaluate(Xtest, Ytest, verbose=0)          

print("Test score:", score[0])

print("Test accuracy:", score[1])
Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xboth,labelCat, test_size = 0.25)



model.fit(Xtrain, Ytrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(Xtest, Ytest))



score = model.evaluate(Xtest, Ytest, verbose=0)          

print("Test score:", score[0])

print("Test accuracy:", score[1])