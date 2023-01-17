import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
Y = train['label']
y_train = np_utils.to_categorical(Y)
train.drop('label',axis=1,inplace=True)
X_train = train/255
X_test = test/255
X_train = X_train.values
X_test = X_test.values
num_classes = y_train.shape[1]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
Pred = model.predict(X_test)
Pred.shape
y_pred = Pred.argmax(axis=1)
ImageID = np.arange(len(y_pred))+1
Out = pd.DataFrame([ImageID,y_pred]).T
Out.rename(columns = {0:'ImageId', 1:'Label'})
#Out
Out.to_csv('MNIST_Prediction.csv', header =  ['ImageId', 'Label' ], index = None) 
