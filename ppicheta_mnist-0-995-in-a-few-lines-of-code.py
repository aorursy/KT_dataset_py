%matplotlib inline



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras
train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = "submission.csv"



raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

x_train, y_train = raw_data[:, 1:], raw_data[:, 0]



x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")/255

y_train = keras.utils.to_categorical(y_train)
model = keras.models.Sequential([

    keras.layers.Conv2D(32, kernel_size=3, activation='relu',

                        input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same',

                        activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(rate=0.4),



    keras.layers.Conv2D(64, kernel_size=3, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(64, kernel_size=3, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same',

                        activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(rate=0.4),



    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(rate=0.4),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=["accuracy"])
from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage.interpolation import map_coordinates





# thanks to https://www.kaggle.com/babbler/mnist-data-augmentation-with-elastic-distortion

def elastic_transform(image, alpha_range, sigma, random_state=None):



    random_state = np.random.RandomState(random_state)



    if np.isscalar(alpha_range):

        alpha = alpha_range

    else:

        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])



    shape = image.shape



    # convolve uniform(-1, 1) values with input response of Gaussian function

    # scale them with alpha parameter

    dx = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma) * alpha

    dy = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma) * alpha



    # prepare coordinate displacements

    x, y, z = np.meshgrid(np.arange(shape[0]),

                          np.arange(shape[1]),

                          np.arange(shape[2]), indexing='ij')

    indices = (np.reshape(x + dx, (-1, 1)),

               np.reshape(y + dy, (-1, 1)),

               np.reshape(z, (-1, 1)))



    # map image to distorted new coordinates by interpolation

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
datagen = keras.preprocessing.image.ImageDataGenerator(

    zoom_range=0.0,

    height_shift_range=2,

    width_shift_range=2,

    preprocessing_function=lambda x: elastic_transform(x, alpha_range=[8, 10], sigma=3))



datagen.fit(x_train)
batch_size = 32

epochs = 30



learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)



model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 

                    epochs=epochs, verbose=2,

                    callbacks=[learning_rate_reduction],

                    steps_per_epoch=x_train.shape[0] // batch_size)
raw_data_test = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

x_test = raw_data_test.reshape(-1, 28, 28, 1).astype("float32")/255
results = model.predict_classes(x_test)

results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1, x_test.shape[0] + 1), name='ImageId'), results], axis=1)

submission.to_csv(output_file, index=False)