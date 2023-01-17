from tensorflow import keras

from tensorflow.keras import layers



model = keras.Sequential([

    layers.Conv2D(filters=64, kernel_size=3), # activation is None

    layers.MaxPool2D(pool_size=2),

    # More layers follow

])


import tensorflow as tf

import matplotlib.pyplot as plt

import warnings



plt.rc('figure', autolayout=True)

plt.rc('axes', labelweight='bold', labelsize='large',

       titleweight='bold', titlesize=18, titlepad=10)

plt.rc('image', cmap='magma')

warnings.filterwarnings("ignore") # to clean up output cells



# Read image

image_path = '../input/computer-vision-resources/car_feature.jpg'

image = tf.io.read_file(image_path)

image = tf.io.decode_jpeg(image)



# Define kernel

kernel = tf.constant([

    [-1, -1, -1],

    [-1,  8, -1],

    [-1, -1, -1],

], dtype=tf.float32)



# Reformat for batch compatibility.

image = tf.image.convert_image_dtype(image, dtype=tf.float32)

image = tf.expand_dims(image, axis=0)

kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])



# Filter step

image_filter = tf.nn.conv2d(

    input=image,

    filters=kernel,

    # we'll talk about these two in the next lesson!

    strides=1,

    padding='SAME'

)



# Detect step

image_detect = tf.nn.relu(image_filter)



# Show what we have so far

plt.figure(figsize=(12, 6))

plt.subplot(131)

plt.imshow(tf.squeeze(image), cmap='gray')

plt.axis('off')

plt.title('Input')

plt.subplot(132)

plt.imshow(tf.squeeze(image_filter))

plt.axis('off')

plt.title('Filter')

plt.subplot(133)

plt.imshow(tf.squeeze(image_detect))

plt.axis('off')

plt.title('Detect')

plt.show();
import tensorflow as tf



image_condense = tf.nn.pool(

    input=image_detect, # image in the Detect step above

    window_shape=(2, 2),

    pooling_type='MAX',

    # we'll see what these do in the next lesson!

    strides=(2, 2),

    padding='SAME',

)



plt.figure(figsize=(6, 6))

plt.imshow(tf.squeeze(image_condense))

plt.axis('off')

plt.show();