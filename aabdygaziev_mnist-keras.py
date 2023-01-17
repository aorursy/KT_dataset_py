import numpy as np

import keras

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
# let's upload train data

train_data_file = open('../input/mnist-train/mnist_train.csv','r')

train_data_list = train_data_file.readlines()

train_data_file.close()



# # let's upload test data

test_data_file = open('../input/mnist-ml-crash-course/mnist_test.csv','r')

test_data_list = test_data_file.readlines()

test_data_file.close()
print('Number of training examples: ',len(train_data_list))

print('Number of test examples: ',len(test_data_list))
# y - targets

# X - features

y_train = []

X_train = []



for record in range(len(train_data_list)):

    y_train.append(train_data_list[record][0])

    values = train_data_list[record].split(',')

    X_train.append(values[1:])



y_test = []

X_test = []



for record in range(len(test_data_list)):

    y_test.append(test_data_list[record][0])

    values = test_data_list[record].split(',')

    X_test.append(values[1:])
# converting to numpy array

y_train = np.asfarray(y_train)

X_train = np.asfarray(X_train)



y_test = np.asfarray(y_test)

X_test = np.asfarray(X_test)
train_images = X_train.reshape((-1, 784))

test_images = X_test.reshape((-1, 784))



# check the shapes

print('y_train shape:',y_train.shape)

print('X_train shape: ',X_train.shape)



print('X_test shape: ',y_test.shape)

print('X_test shape: ',X_test.shape)
# Normalize the images.

train_images = ((train_images / 255) * 0.99) + 0.01

test_images = ((test_images / 255) * 0.99) + 0.01
# instantiate model

from keras.models import Sequential

from keras.layers import Dense



model = Sequential([

    Dense(784,activation='relu',input_shape=(784,)),

    Dense(200,activation='relu',kernel_regularizer='l2',bias_regularizer='l2'),

    Dense(200,activation='relu',kernel_regularizer='l2',bias_regularizer='l2'),

    Dense(10,activation='softmax')

]);
model.compile(

    optimizer=keras.optimizers.Adam(learning_rate=0.01),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)
from keras.utils import to_categorical



model.fit(

    x=train_images, #train data-set

    y=to_categorical(y_train), #labels

    epochs=5,

    batch_size=32,

    validation_split=0.15

)
model.evaluate(

    test_images,

    to_categorical(y_test)

)
model = Sequential([

    Dense(64,activation='relu'),

    Dense(64,activation='relu'),

    Dense(10,activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)





from keras.utils import to_categorical



model.fit(

    x=train_images, #train data-set

    y=to_categorical(y_train), #labels

    epochs=10,

    batch_size=32

)



print('test accuracy: ')



model.evaluate(

  test_images,

  to_categorical(y_test)

)

# more layers

model = Sequential([

    Dense(64,activation='relu'),

    Dense(64,activation='relu'),

    Dense(100,activation='relu'),

    Dense(100,activation='relu'),

    Dense(10,activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)





from keras.utils import to_categorical



model.fit(

    x=train_images, #train data-set

    y=to_categorical(y_train), #labels

    epochs=5,

    batch_size=32

)





model.evaluate(

  test_images,

  to_categorical(y_test)

)



model = Sequential([

    Dense(64,activation='sigmoid'),

    Dense(64,activation='sigmoid'),

    Dense(10,activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)





from keras.utils import to_categorical



model.fit(

    x=train_images, #train data-set

    y=to_categorical(y_train), #labels

    epochs=5,

    batch_size=32

)



print('test accuracy: ')



model.evaluate(

  test_images,

  to_categorical(y_test)

)