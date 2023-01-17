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
import keras
train = pd.read_csv('../input/train.csv')
labels = train.iloc[:,0].values.astype('int32')
#y = train.iloc[:,0].values.astype('int32')
train_X = train.iloc[:,1:].values.astype('float32')
test = (pd.read_csv('../input/test.csv').values).astype("float32")

num_classes=10
labels=keras.utils.to_categorical(labels, num_classes)
train_X.shape
labels.shape
test.shape
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
batch_size = 128
num_classes = 10
epochs = 80
img_rows,img_cols = 28,28
train_X/=255
test/=255
from keras import backend as K
K.image_data_format()
input_shape = (img_rows,img_cols,1)

#trying out a new model

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu',input_shape = input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols,1)
adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

model.fit(train_X, labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
validation_split=0.1)
test = test.reshape(test.shape[0], img_rows, img_cols,1)
preds = model.predict_classes(test, verbose=0)
def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "mnist-cnn.csv")
!ls

