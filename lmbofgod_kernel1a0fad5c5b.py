import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Xtrain = np.load('/kaggle/input/data-and-model/npy_files/x_train.npy')

Ytrain = np.load('/kaggle/input/data-and-model/npy_files/y_train.npy')
Ytrain[0]
Xtest = np.load('/kaggle/input/data-and-model/npy_files/x_test.npy')

Ytest = np.load('/kaggle/input/data-and-model/npy_files/y_test.npy')
import keras

from keras.applications.nasnet import NASNetMobile 

from keras.layers import Dense

from keras.models import Model

from keras.optimizers import Adam

from keras.utils import to_categorical
model = NASNetMobile(input_shape=Xtrain[0].shape, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model.summary()
print(len(model.layers))

model.layers.pop()

dense_layer = Dense(4,activation='softmax')(model.layers[-1].output)

new_model = Model(model.input,dense_layer)
new_model.summary()
opt = Adam(lr=0.0002, beta_1=0.5)

new_model.compile(loss='binary_crossentropy', optimizer=opt,metrics =['accuracy'])
Ytest
Ytrain[0:50]

Ytrain = to_categorical(Ytrain, num_classes=4, dtype='float32')
Ytest = to_categorical(Ytest, num_classes=4, dtype='float32')
Ytest
new_model.fit(x=Xtrain, y=Ytrain, epochs=200,validation_data=(Xtest,Ytest))