import sys

import numpy

numpy.set_printoptions(threshold=sys.maxsize)

from sklearn.model_selection import train_test_split

import pandas as pd

import keras

import tensorflow

from keras.models import Sequential

from keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.utils.np_utils import to_categorical
train= pd.read_csv('../input/Kannada-MNIST/train.csv')

test= pd.read_csv('../input/Kannada-MNIST/test.csv', index_col='id')

print('Train shape: ', train.shape)

print('Train missing vals: ', train.isna().sum().sum())

print('Test shape: ', test.shape)

print('Test missing vals: ', test.isna().sum().sum())
# x_train is all rows & columns of train except 1st column

x_train= train.iloc[:, 1:].values

# y_train is only 1st column of train

y_train= train.iloc[:, 0].values

# x_test is all rows and columns of test

x_test= test.iloc[:, :].values
# reshape data to 60000 single layered 28x28 matrices

x_train= x_train.reshape(train.shape[0], 28, 28, 1)

x_test= x_test.reshape(test.shape[0], 28, 28, 1)

# instead of 0123 convert to 1000000000 0100000000 0010000000 0001000000

y_train= to_categorical(y_train, 10)

# splitting training data into training and validation sets

x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train, test_size= 0.2, random_state= 42)
'''# setting the params

num_classes= 10

lr=  0.001

batch_size= 1024

#creating sequential model

model=  Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size= 3, activation= 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size= 3, activation= 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size= 5, padding= 'same', activation= 'relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(256, kernel_size= 3, activation= 'relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.summary()



optimizer = RMSprop(lr=lr)

model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics= ['accuracy'])'''
# setting the params

num_classes= 10

lr=  0.001

batch_size= 1024

#creating sequential model

model=  Sequential()



model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=3, padding='same',  activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=5, padding='same', strides= 2, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Conv2D(64, kernel_size= 3, padding='same', activation= 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size= 3, padding='same', activation= 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size= 5, padding= 'same', strides= 2, activation= 'relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.summary()



optimizer = RMSprop(lr=lr)

model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics= ['accuracy'])
img_gen= ImageDataGenerator(rotation_range= 5, zoom_range= 0.5, horizontal_flip= 0, vertical_flip= 0,

                           width_shift_range= 2, height_shift_range= 2, rescale= 1/255)



valid_img_gen = ImageDataGenerator(rescale=1./255.)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=200, verbose=1, factor=0.2)



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)



model.fit_generator(img_gen.flow(x_train, y_train, batch_size= batch_size), steps_per_epoch= 100, epochs= 100,

                   validation_data= valid_img_gen.flow(x_valid, y_valid, batch_size= batch_size), validation_steps= 50,

                    callbacks=[learning_rate_reduction, es], verbose= 1)
pred= model.predict_classes(x_test/255)

sub= pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

sub['label']= pred

sub.to_csv('submission.csv', index= False)