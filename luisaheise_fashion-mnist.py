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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
pure = np.load("../input/train_images_pure.npy")
plt.subplot(221)
plt.imshow(pure[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(pure[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(pure[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(pure[3], cmap=plt.get_cmap('gray'))
plt.show()
rotated = np.load("../input/train_images_rotated.npy")


noisy = np.load("../input/train_images_noisy.npy")


both = np.load("../input/train_images_both.npy")

plt.subplot(221)
plt.imshow(rotated[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(rotated[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(rotated[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(rotated[3], cmap=plt.get_cmap('gray'))
plt.show()
plt.subplot(221)
plt.imshow(noisy[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(noisy[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(noisy[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(noisy[3], cmap=plt.get_cmap('gray'))
plt.show()
plt.subplot(221)
plt.imshow(both[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(both[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(both[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(both[3], cmap=plt.get_cmap('gray'))
plt.show()
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping
#Seed constante
seed = 7
np.random.seed(seed)
Xpure = pure.reshape(pure.shape[0], 28, 28, 1).astype('float32')
Xnoisy = noisy.reshape(noisy.shape[0], 28, 28, 1).astype('float32')
Xrotated = rotated.reshape(rotated.shape[0], 28, 28, 1).astype('float32')
Xboth = both.reshape(both.shape[0], 28, 28, 1).astype('float32')
Xpure.shape
from keras.utils import np_utils
trainLabels = pd.read_csv("../input/train_labels.csv", header=0, index_col=0, na_values="?")
trainLabels = np_utils.to_categorical(trainLabels)
numberClasses = trainLabels.shape[1]
Xpure/= 255
Xnoisy/= 255
Xrotated/= 255
Xboth/= 255
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xpure,trainLabels, test_size = 0.25)
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(numberClasses, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
XRtrain,XRtest,YRtrain,YRtest = train_test_split(Xboth,trainLabels, test_size = 0.25)
model.fit(XRtrain, YRtrain, validation_data=(XRtest, YRtest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
def Baseline_model2():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(numberClasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
modelo2 = Baseline_model2()
modelo2.summary()
modelo2.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
#Estabelecendo as bases com ruído/rotações
XStrain,XStest,YStrain,YStest = train_test_split(Xrotated,trainLabels, test_size = 0.25)
XNtrain,XNtest,YNtrain,YNtest = train_test_split(Xnoisy,trainLabels, test_size = 0.25)
modelo2.fit(Xtrain, Ytrain, validation_data=(XStest, YStest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(Xtrain, Ytrain, validation_data=(XNtest, YNtest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(Xtrain, Ytrain, validation_data=(XRtest, YRtest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(XStrain, YStrain, validation_data=(XStest, YStest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(XNtrain, YNtrain, validation_data=(XNtest, YNtest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(XRtrain, YRtrain, validation_data=(XRtest, YRtest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(XRtrain, YRtrain, validation_data=(Xtest, Ytest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(XRtrain, YRtrain, validation_data=(XNtest, YNtest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
modelo2.fit(XRtrain, YRtrain, validation_data=(XStest, YStest), epochs=20, batch_size=200, verbose=1, callbacks = callbacks)
scores = model.evaluate(Xtest, Ytest, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))