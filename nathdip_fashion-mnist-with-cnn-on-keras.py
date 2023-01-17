import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Importing the data set
train = pd.read_csv('../input/fashion-mnist_train.csv')

test =  pd.read_csv('../input/fashion-mnist_test.csv')
train.head()
X_train = train.drop(['label'], axis = 1)

y_train = train['label']

X_test = test.drop(['label'], axis = 1)

y_test = test['label']
X_train = X_train.as_matrix()

y_train = y_train.as_matrix()

X_test = X_test.as_matrix()

y_test = y_test.as_matrix()
#Plotting one of the images

explore_image_no = 500
plt.imshow(X_train[explore_image_no].reshape((28, 28)), cmap = 'gray')
X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))

X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255

num_classes = 10

print('Shape of training set: ', X_train.shape)

print('Shape of test set: ', X_test.shape)
import keras

#making the y labels catgeorical

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

print('Training label size: ', y_train.shape)

print('Test label size: ', y_test.shape)
#Building the model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
batch_size = 128

epochs = 12



# input image dimensions

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 padding = 'SAME',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding = 'SAME', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_test, y_test))