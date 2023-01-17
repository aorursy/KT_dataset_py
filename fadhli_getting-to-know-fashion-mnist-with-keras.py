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
# getting the test and train dataset

fashion_train = pd.read_csv('../input/fashion-mnist_train.csv')

fashion_test = pd.read_csv('../input/fashion-mnist_test.csv')
# checking on the training data

fashion_train.head()
# checking for any null values

print(fashion_train.isnull().sum().sum())

print(fashion_test.isnull().sum().sum())
# import to_categorical (one hot encoding)



from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



# train data

X = fashion_train.drop(['label'], axis=1).values

y = to_categorical(fashion_train['label'].values)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# test data

X_test = fashion_test.drop(['label'], axis=1).values

y_test = to_categorical(fashion_test['label'].values)
# reshape the data

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_test /= 255

X_val /= 255
# import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization



batch_size = 256

num_classes = 10

epochs = 50



#input image dimensions

img_rows, img_cols = 28, 28



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 kernel_initializer='he_normal',

                 input_shape=input_shape))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])

print('Test accuracy:', score[1])