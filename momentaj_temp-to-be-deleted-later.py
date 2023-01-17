import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

K.set_image_dim_ordering('th')
train_data = pd.read_csv("../input/train.csv").values

test_data  = pd.read_csv("../input/test.csv").values
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)

train, test = train_test_split(train_data, test_size = 0.2)



X_train = train[:,1:]

X_train = X_train.astype(np.float)

y_train = train[:,:1].ravel()

X_test = test[:,1:]

X_test = X_test.astype(np.float)

y_test = test[:,:1].ravel()
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255

test_data = test_data / 255
# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

num_pixels = 28 * 28
# fix random seed for reproducibility

np.random.seed(seed)



def larger_model():

	# create model

	model = Sequential()

	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(15, 3, 3, activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model



# build the model

model = larger_model()



# fix random seed for reproducibility

np.random.seed(seed)



# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)