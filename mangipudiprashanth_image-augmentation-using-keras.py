%matplotlib inline



import os

import numpy as np

import tensorflow as tf



from PIL import Image

from matplotlib import pyplot as plt



print('Using TensorFlow', tf.__version__)
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    rotation_range=40

)
image_path = '../input/data-aug/images/train/cat/cat.jpg'



plt.imshow(plt.imread(image_path));
x, y = next(generator.flow_from_directory('../input/data-aug/images', batch_size=1))

plt.imshow(x[0].astype('uint8'));
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    width_shift_range=[-40, -20, 0, 20, 40],

    height_shift_range=[-50,50]

)
x, y = next(generator.flow_from_directory('../input/data-aug/images', batch_size=1))

plt.imshow(x[0].astype('uint8'));
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    brightness_range=(0., 2.)

)



x, y = next(generator.flow_from_directory('../input/data-aug/images', batch_size=1))

plt.imshow(x[0].astype('uint8'));
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    shear_range=45

)



x, y = next(generator.flow_from_directory('../input/data-aug/images', batch_size=1))

plt.imshow(x[0].astype('uint8'));
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    zoom_range=0.5

)



x, y = next(generator.flow_from_directory('../input/data-aug/images', batch_size=1))

plt.imshow(x[0].astype('uint8'));
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    channel_shift_range=100

)



x, y = next(generator.flow_from_directory('../input/data-aug/images', batch_size=1))

plt.imshow(x[0].astype('uint8'));
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    horizontal_flip=True,

    vertical_flip=True

)



x, y = next(generator.flow_from_directory('../input/data-aug/images', batch_size=1))

plt.imshow(x[0].astype('uint8'));
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



generator = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True

)



generator.fit(x_train)
x, y = next(generator.flow(x_train, y_train, batch_size=1))

print(x.mean(), x.std(), y)

print(x_train.mean())
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    samplewise_center=True,

    samplewise_std_normalization=True

)



x, y = next(generator.flow(x_train, y_train, batch_size=1))

print(x.mean(), x.std(), y)
generator = tf.keras.preprocessing.image.ImageDataGenerator(

    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,

    rescale=1.

)
x, y = next(generator.flow(x_train, y_train, batch_size=1))
print(x.mean(), x.std(), y)
model = tf.keras.models.Sequential([

    tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=(32, 32, 3), pooling='avg'),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
_ = model.fit(

    generator.flow(x_train, y_train, batch_size=32),

    steps_per_epoch=10, epochs=1

)