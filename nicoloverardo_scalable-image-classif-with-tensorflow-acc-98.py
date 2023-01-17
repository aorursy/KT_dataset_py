import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
DATASET_PATH = "/kaggle/input/turkish-lira-banknote-dataset"
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
IMG_WIDTH = 64
IMG_HEIGHT = 64

NUM_WORKERS = 1
PER_WORKER_BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = PER_WORKER_BATCH_SIZE * NUM_WORKERS
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  DATASET_PATH,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_WIDTH, IMG_HEIGHT),
  label_mode='categorical',
  batch_size=GLOBAL_BATCH_SIZE)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  DATASET_PATH,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_WIDTH, IMG_HEIGHT),
  label_mode='categorical',
  batch_size=GLOBAL_BATCH_SIZE)
class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.map(scale, num_parallel_calls=AUTOTUNE).repeat().cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(scale, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
dist_dataset = strategy.experimental_distribute_dataset(train_ds)
with strategy.scope():
    model = Sequential()

    # VGG Blocks
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense layers
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
es = EarlyStopping(monitor='loss', verbose=1, mode='min', patience = 2, min_delta=0.01)
history = model.fit(dist_dataset,
            epochs=15,
            steps_per_epoch = 75,
            callbacks=[es])
model.evaluate(test_ds)