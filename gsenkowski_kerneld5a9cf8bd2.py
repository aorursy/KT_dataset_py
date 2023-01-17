# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras
from keras.models import Model
from keras.layers import Dense, AlphaDropout, Lambda, Flatten, Input, Conv2D, MaxPool2D, GaussianNoise
from scipy import ndimage
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#from the first tutorial https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:42000,1:]
labels = labeled_images.iloc[0:42000,:1]

#from the second tutorial https://www.kaggle.com/poonaml/deep-neural-network-keras-way
X_train = (images.iloc[:,0:].values).astype('float32') # all pixel values
y_train = labels.iloc[:,0].values.astype('int32') # only labels i.e targets digits


X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_train = X_train / 255

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)


# This converts the labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train)

#from the second tutorial https://www.kaggle.com/poonaml/deep-neural-network-keras-way
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px

inputs = Input(shape=(28, 28, 1, ))
flatten = Flatten()(inputs)
lmbda = Lambda(standardize)(flatten)
noise = GaussianNoise(0.75)(lmbda)
dense1 = Dense(2500, activation='relu', kernel_initializer='lecun_normal')(noise)
dropout1 = AlphaDropout(0.25)(dense1)
dense2 = Dense(2000, activation='relu', kernel_initializer='lecun_normal')(dropout1)
dropout2 = AlphaDropout(0.25)(dense2)
dense3 = Dense(1500, activation='relu', kernel_initializer='lecun_normal')(dropout2)
dropout3 = AlphaDropout(0.25)(dense3)
dense4 = Dense(1000, activation='relu', kernel_initializer='lecun_normal')(dropout3)
dropout4 = AlphaDropout(0.25)(dense4)
dense5 = Dense(500, activation='relu', kernel_initializer='lecun_normal')(dropout4)
dropout5 = AlphaDropout(0.25)(dense5)
predictions = Dense(10, activation='softmax')(dropout5)
model = Model(inputs=inputs, outputs=predictions)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(momentum=0.9, nesterov=True),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=16,
          epochs=50,
          verbose=1,
          validation_data=None)

score = model.evaluate(X_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Any results you write to the current directory are saved as output.


test = pd.read_csv("../input/test.csv")
test = test.values.astype('float32')
test = test.reshape(test.shape[0], 28, 28)

test /= 255

test = test.reshape(test.shape[0], 28, 28, 1)

#from the second tutorial https://www.kaggle.com/poonaml/deep-neural-network-keras-way

predictions = model.predict(test, verbose=0)

predictions = [np.argmax(prediction) for prediction in predictions]

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("results.csv", index=False, header=True)
