!pip -q install tensorflow-gpu==2.0.0-beta1

!pip -q install tensorflow-addons
import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_addons as tfa



from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.layers import (

    BatchNormalization,

    Dense,

    Dropout,

    Flatten,

    Conv2D,

    MaxPool2D,

)



import matplotlib.pyplot as plt



AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 28

IMG_WIDTH = 28

BATCH_SIZE = 128

BUFFER_SIZE = 1000
def reshape(image):

    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 1])

    return image



def normalize(image):

    image = tf.cast(image, tf.float32)

    image = (image / 127.5) - 1

    return image



def random_crop(image):

    cropped_image = tf.image.random_crop(

        image, size=[IMG_HEIGHT, IMG_WIDTH, 1])

    return cropped_image



def random_rotate(image):

    rotate_angles = tf.random.normal([], stddev=0.2)

    image = tfa.image.rotate(image, rotate_angles)

    return image



def random_jitter(image):

    # resizing to 30 x 30 x 1

    image = tf.image.resize(image, [30, 30],

                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)



    # randomly cropping to 28 x 28 x 1

    image = random_crop(image)

    return image



def preprocess_train(*features):

    label = features[0]

    image = tf.stack(features[1:], axis=-1)

    

    image = reshape(image)

    image = random_rotate(image)

    image = random_jitter(image)

    image = normalize(image)

    return image, label



def preprocess_eval(*features):

    label = features[0]

    image = tf.stack(features[1:], axis=-1)

    

    image = reshape(image)

    image = normalize(image)

    return image, label



def preprocess_test(*features):

    image = tf.stack(features, axis=-1)



    image = reshape(image)

    image = normalize(image)

    return image
ds = tf.data.experimental.CsvDataset(

    "../input/train.csv",

    [tf.int64] * 785,

    header=True,

)



ds_test = tf.data.experimental.CsvDataset(

    "../input/test.csv",

    [tf.int64] * 784,

    header=True,

)



num_eval = int(42000 * 0.1)

ds_eval = ds.take(num_eval)

ds_train = ds.skip(num_eval)



ds_train = ds_train.map(preprocess_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

ds_eval = ds_eval.map(preprocess_eval, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

ds_test = ds_test.map(preprocess_test, num_parallel_calls=AUTOTUNE).batch(32)
NUM_SAMPLES = 6



def preprocess_sample(*features):

    image = tf.stack(features[1:], axis=-1)

    image = reshape(image)

    return image



sample_images = next(iter(ds.map(preprocess_sample).batch(NUM_SAMPLES)))



fig=plt.figure(figsize=(10, 10))

fig.tight_layout()



for i, sample_image in enumerate(sample_images, start=1):

    fig.add_subplot(3, NUM_SAMPLES, i)

    plt.imshow(sample_image[:, :, 0], cmap='gray_r')



    fig.add_subplot(3, NUM_SAMPLES, NUM_SAMPLES + i)

    plt.imshow(random_jitter(sample_image)[:, :, 0], cmap='gray_r')



    fig.add_subplot(3, NUM_SAMPLES, 2 * NUM_SAMPLES + i)

    plt.imshow(random_rotate(sample_image)[:, :, 0], cmap='gray_r')

plt.show()
model = tf.keras.models.Sequential(

    [

        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),

        BatchNormalization(),

        Conv2D(32, (3, 3), activation="relu"),

        BatchNormalization(),

        Conv2D(32, (5, 5), activation="relu", strides=2, padding="same"),

        BatchNormalization(),

        Dropout(0.4),

        

        Conv2D(64, (3, 3), activation="relu"),

        BatchNormalization(),

        Conv2D(64, (3, 3), activation="relu"),

        BatchNormalization(),

        Conv2D(64, (5, 5), activation="relu", strides=2, padding="same"),

        BatchNormalization(),

        Dropout(0.4),

        

        Conv2D(128, (4, 4), activation="relu"),

        BatchNormalization(),

        Flatten(),

        Dropout(0.4),

        Dense(10, activation="softmax"),

    ]

)
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),

              loss="sparse_categorical_crossentropy",

              metrics=["sparse_categorical_accuracy"])
lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1, factor=0.5, min_lr=1e-5)



history = model.fit(ds_train, epochs=30, validation_data=ds_eval, callbacks=[lr_reduction])
predictions = model.predict(ds_test)
results = np.argmax(predictions, axis=1)

results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), results], axis=1)

submission.to_csv("cnn_submission.csv", index=False)