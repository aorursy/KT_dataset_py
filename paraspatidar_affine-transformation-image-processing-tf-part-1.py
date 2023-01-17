import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.image as mpimg
file = tf.keras.utils.get_file(

    "file.jpg",

    "https://pixlr.com/photo/image-design-11-1-pw.jpg")

img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

plt.imshow(img)
img = np.array(img)
type(img)
plt.imshow(img)
transformation = tf.keras.preprocessing.image.apply_affine_transform(

    img,

    theta=270

)

plt.imshow(transformation)
transformation = tf.keras.preprocessing.image.apply_affine_transform(

    img,

    shear=50

)

plt.imshow(transformation)
transformation = tf.keras.preprocessing.image.apply_affine_transform(

    img,

    tx=30,

    ty=30

)

plt.imshow(transformation)
transformation = tf.keras.preprocessing.image.apply_affine_transform(

    img,

    zx=0.5,

    zy= 0.5

)

plt.imshow(transformation)

transformation = tf.keras.preprocessing.image.apply_affine_transform(

    img,

   row_axis=0,

    col_axis=1,

    channel_axis=2,

    fill_mode='nearest',

    cval=0.5,

    order=1

)

plt.imshow(transformation)
