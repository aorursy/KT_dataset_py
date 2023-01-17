# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.utils import to_categorical

data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')



X_train = np.array(data_train.iloc[:, 1:])

X_test = np.array(data_test.iloc[:, 1:])

y_train = to_categorical(np.array(data_train.iloc[:, 0]))

y_test = to_categorical(np.array(data_test.iloc[:, 0]))
img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D



input_shape = (img_rows, img_cols, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',

          loss='categorical_crossentropy',

          metrics=['accuracy'])
model.fit(X_train, y_train,

          batch_size=24,

          epochs=10,

          verbose=1,

          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('accuracy: ',score[1])

print('loss: ',score[0])