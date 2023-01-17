# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
nb_classes = 10
train_df = pd.read_csv("../input/train.csv")
y_train = []
data = []
for i in range(train_df.shape[0]):
    y_train.append(train_df.iloc[i, 0])
    data.append(train_df.iloc[i, 1:].values)

X_train = np.array(data).astype('float32')/255

test_df = pd.read_csv("../input/test.csv")
y_test = []
data = []
for i in range(test_df.shape[0]):
    y_test.append(test_df.iloc[i, 0])
    data.append(test_df.iloc[i, 1:].values)

X_test = np.array(data).astype('float32')/255


# one-hot representation
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(128, input_dim=input_unit_size, init='glorot_uniform'))
model.add(Activation("relu"))
model.add(Dropout(p=0.2))
model.add(Dense(nb_classes, init='glorot_uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])
nb_epoch = 10
result = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=256, verbose=2, validation_split=0.2)