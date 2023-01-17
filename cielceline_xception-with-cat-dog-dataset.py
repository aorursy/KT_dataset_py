import tensorflow as tf

import numpy as np

import glob

import os

from matplotlib import pyplot as plt

%matplotlib inline



keras = tf.keras

layers = tf.keras.layers
def load_preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [256, 256])

    image = tf.cast(image, tf.float32)

    image = image/255

    

    return image, label
path = glob.glob('../input/cat-and-dog/training_set/training_set/*/*.jpg')

path = path[:2000] + path[-2000:]



label = [int(p.split('/')[5]=='cats') for p in path]



train_ds = tf.data.Dataset.from_tensor_slices((path, label))



AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)



for img, label in train_ds.take(1):

    plt.imshow(img)



train_ds = train_ds.shuffle(4000).repeat().batch(32)
test_path = glob.glob('../input/cat-and-dog/test_set/test_set/*/*.jpg')

test_path = test_path[500:1000] + test_path[1500:2000]



test_label = [int(p.split('/')[5]=='cats') for p in test_path]



test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_label))

test_ds = test_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)

    

test_ds = test_ds.batch(32).repeat()
conv_base = keras.applications.xception.Xception(weights='imagenet',

                                                 include_top=False,

                                                 input_shape=(256, 256, 3),

                                                 pooling='avg')

conv_base.trainable = False
conv_base.summary()
model = keras.Sequential()

model.add(conv_base)

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(train_ds, steps_per_epoch=4000//32, epochs=5, validation_data=test_ds, validation_steps=1000//32)
conv_base.trainable = True

len(conv_base.layers)
fine_tune_at = -33

for layer in conv_base.layers[:fine_tune_at]:

    conv_base.trainable = False
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005/10),

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(train_ds, steps_per_epoch=4000//32, epochs=7, initial_epoch=5, validation_data=test_ds, validation_steps=1000//32)
history.history.keys()
plt.plot(history.epoch, history.history.get('acc'), label='acc')

plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')

plt.legend()
plt.plot(history.epoch, history.history.get('loss'), label='loss')

plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')

plt.legend()