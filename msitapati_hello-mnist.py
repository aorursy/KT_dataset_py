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
df_test = pd.read_csv('../input/test.csv')
df_train = pd.read_csv('../input/train.csv')
(X_train, y_train) = df_train.drop('label', axis=1), df_train['label']
X_train.head(), y_train.head()
X_train.shape, y_train.shape
X_test = pd.read_csv('../input/test.csv')



df_test['label'] = 0

y_test = df_test['label']
X_test.head(), y_test.head()
X_test.shape, y_test.shape
image_height, image_width = 28, 28
X_train = X_train.values.reshape(42000, image_height * image_width)



print(X_train.shape)
X_test = X_test.values.reshape(28000, image_height * image_width)



print(X_test.shape)
# make into float numbers

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



# hot encoding from 0 to 1

X_train /= 255.0

X_test /= 255.0



print(X_train[0])
print(y_train.shape)
from keras.utils.np_utils import to_categorical



y_train = to_categorical(y_train, 10)

print(y_train.shape)

y_test = to_categorical(y_test, 10)

print(y_test.shape)
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()
# first layer

model.add(Dense(512, activation='relu', input_shape=(784,)))



#second layer

model.add(Dense(512, activation='relu'))



# third layer

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',                 # adam is the got-to optimizer in general

              loss='categorical_crossentropy',  # 10 classes/bins. this function allows for that

              metrics=['accuracy']              # accuracy

             )
model.summary()
from keras.callbacks import TensorBoard
tboard = TensorBoard(log_dir='./output', 

                     histogram_freq=5, 

                     write_graph=True, 

                     write_images=True

                    )
history = model.fit(X_train, y_train, 

                    epochs=8, 

                    validation_data=(X_test, y_test),

                    validation_split=1/6, 

                    callbacks=[tboard]

                   )
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(history.history['acc'])      # accuracy of training set

plt.plot(history.history['val_acc'])  # accuracy of testing set
plt.plot(history.history['loss'])     # loss score
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
score = model.evaluate(X_test, y_test)
score