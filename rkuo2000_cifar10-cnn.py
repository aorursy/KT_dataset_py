# Import Libraries

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.datasets import cifar10 # use Keras built-in CIFAR10 dataset

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
# set parameters

batch_size = 32

num_classes = 10
# Split dataset into train and test:

# Load CIFAR10 data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
# Prepare train & test Data

y_train = to_categorical(y_train, num_classes) # one-hot encoding

y_test = to_categorical(y_test, num_classes)   # one-hot encoding



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255.0 # convert to floating point

x_test /= 255.0  # convert to floating point
# Build Model

model = keras.models.Sequential()

# 1st Conv block

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

# 2nd Conv block

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

# Fully-Connected layer

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.summary()
# Compile Model

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# Data Generator

# This will do preprocessing and realtime data augmentation:

datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    zca_epsilon=1e-06,  # epsilon for ZCA whitening

    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

    # randomly shift images horizontally (fraction of total width)

    width_shift_range=0.1,

    # randomly shift images vertically (fraction of total height)

    height_shift_range=0.1,

    shear_range=0.,  # set range for random shear

    zoom_range=0.,  # set range for random zoom

    channel_shift_range=0.,  # set range for random channel shifts

    # set mode for filling points outside the input boundaries

    fill_mode='nearest',

    cval=0.,  # value used for fill_mode = "constant"

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False,  # randomly flip images

    # set rescaling factor (applied before any other transformation)

    rescale=None,

    # set function that will be applied on each input

    preprocessing_function=None,

    # image data format, either "channels_first" or "channels_last"

    data_format=None,

    # fraction of images reserved for validation (strictly between 0 and 1)

    validation_split=0.0)
# Compute quantities required for feature-wise normalization

# (std, mean, and principal components if ZCA whitening is applied).

datagen.fit(x_train)
# Train Model 

num_epochs=10

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=num_epochs, validation_data=(x_test, y_test))
# Evaluate Model

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
# Save Model

#model.save('cifar10.h5')