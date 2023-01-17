import numpy

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)
#Import some basic modules

import numpy as np

import pandas as pd
# Import training data

train = pd.read_csv('../input/train.csv')



print (train.shape)
#Convert dataframe to numpy array, as Neural network will need that as input datatype

train1 = train.iloc[:,1:].values

train1 = train1.astype(np.float)



# divide by 255 to convert pixel values between 0 and 1

train1 = np.multiply(train1, 1.0 / 255.0)



print (train1.shape)
train1 = train1.reshape(train1.shape[0], 28, 28, 1).astype('float32')

print (train1.shape)
#Calculate distinct labels in the data

distinct_labels = np.unique(labels).shape[0]
#Retrieve labels form input data

labels = train[[0]].values.ravel()



print('labels({0})'.format(len(labels)))
#Calculate distinct labels in the data

distinct_labels = np.unique(labels).shape[0]



print('distinct_labels => {0}'.format(distinct_labels))
#Convert Label to binary matrix using using one hot encoding

from keras.utils import np_utils

y = np_utils.to_categorical(labels)

print('labels({0[0]},{0[1]})'.format(y.shape))
#Split into train and test

from sklearn.cross_validation import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train1, y, test_size=0.05, random_state=42)



print('x_train({0[0]},{0[1]})'.format(x_train.shape))

print('x_test({0[0]},{0[1]})'.format(x_test.shape))

print('y_train({0[0]},{0[1]})'.format(y_train.shape))

print('y_test({0[0]},{0[1]})'.format(y_test.shape))
#def baseline_model():

#	# create model

#	model = Sequential()

#	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=( 28, 28, 1), activation='relu'))

#	model.add(MaxPooling2D(pool_size=(2, 2)))

#	model.add(Dropout(0.2))

#	model.add(Flatten())

#	model.add(Dense(128, activation='relu'))

#	model.add(Dense(distinct_labels, activation='softmax'))

#	# Compile model

#	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#	return model