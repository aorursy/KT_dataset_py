import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.applications.resnet50 import ResNet50
train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

dig_mnist = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
train_df.head()
train_data = np.array(train_df.iloc[:, 1:])

val_data = np.array(dig_mnist.iloc[:, 1:])

test_data = np.array(test_df.iloc[:, 1:])



train_labels = to_categorical(train_df.iloc[:, 0])

val_labels = to_categorical(dig_mnist.iloc[:, 0])
rows, cols = 28, 28



train_data = train_data.reshape(train_data.shape[0], rows, cols, 1)

val_data = val_data.reshape(val_data.shape[0], rows, cols, 1)

test_data = test_data.reshape(test_data.shape[0], rows, cols, 1)
train_data = train_data.astype('float32')

val_data = val_data.astype('float32')

test_data = test_data.astype('float32')
train_data /= 255.0

val_data /= 255.0

test_data /= 255.0
batch_size = 64

epochs = 5

input_shape = (rows, cols, 1)
def baseline_model():

    model = Sequential()

    model.add(Conv2D(64, (3,3), strides=1, padding="same", activation='relu', input_shape=input_shape))

    model.add(BatchNormalization())

    #model.add(Dropout(0.15))

    

    model.add(Conv2D(64, (3,3), strides=1, padding="same", activation='relu'))

    model.add(BatchNormalization())

    

    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

    model.add(Dropout(0.15))

    

    model.add(Conv2D(128, (3,3), strides=1, padding="same", activation='relu'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.15))

    

    model.add(Conv2D(128, (3,3), strides=1, padding="same", activation='relu'))

    model.add(BatchNormalization())

    

    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

    model.add(Dropout(0.15))

    

    model.add(Conv2D(256, (3,3), strides=1, padding="same", activation='relu'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.15))

    

    model.add(Conv2D(256, (3,3), strides=1, padding="same", activation='relu'))

    model.add(BatchNormalization())

    

    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

    model.add(Dropout(0.15))

    

    model.add(Flatten())

#     model.add(Dense(128, activation='relu'))

#     model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))

    #model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    return model
model = baseline_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(train_data)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
history = model.fit_generator(datagen.flow(train_data, train_labels, batch_size=batch_size),

                              epochs = epochs, validation_data = (val_data, val_labels),

                              steps_per_epoch=train_data.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
preds = model.predict_classes(test_data)
sample_sub.label = preds
sample_sub.to_csv('SubmissionFile.csv', index=False)