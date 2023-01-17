import pathlib

import os

import numpy as np

import pandas as pb

import tensorflow as tf

import keras

import IPython.display as display

from PIL import Image

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
data_dir = pathlib.Path("../input/cheese/cheese")

image_count = len(list(data_dir.glob('*/*.jpeg')))

image_count
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])

CLASS_NAMES
labels_to_class_names = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))

print(labels_to_class_names)
BATCH_TRAIN_SIZE = 64

IMG_HEIGHT = 224

IMG_WIDTH = 224

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_TRAIN_SIZE)

EPOCHS=12
AUTOTUNE = tf.data.experimental.AUTOTUNE
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))



for f in list_ds.take(5):

    print(f.numpy())
def get_label(file_path):

    # convert the path to a list of path components

    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory

    return parts[-2] == CLASS_NAMES



def decode_img(img):

    # convert the compressed string to a 3D uint8 tensor

    img = tf.image.decode_jpeg(img, channels=3)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.

    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image to the desired size.

    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])



def process_path(file_path):

    label = get_label(file_path)

    # load the raw data from the file as a string

    img = tf.io.read_file(file_path)

    img = decode_img(img)

    return img, label
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(2):

    print("Image shape: ", image.numpy().shape)

    print("Label: ", label.numpy())
train_ds = labeled_ds.take(np.ceil(3233*0.7)) 

test_ds = labeled_ds.take(np.ceil(3233*0.7))



for image, label in train_ds.take(1):

    print("Image shape: ", image.numpy().shape)

    print("Label: ", label.numpy())

    

for image, label in test_ds.take(1):

    print("Image shape: ", image.numpy().shape)

    print("Label: ", label.numpy())

def prepare_for_training(ds, cache=True, shuffle_buffer_size=300):

    # This is a small dataset, only load it once, and keep it in memory.

    # use `.cache(filename)` to cache preprocessing work for datasets that don't

    # fit in memory.

    if cache:

        if isinstance(cache, str):

            ds = ds.cache(cache)

        else:

            ds = ds.cache()



    ds = ds.shuffle(buffer_size=shuffle_buffer_size)



    # Repeat forever

    ds = ds.repeat()



    ds = ds.batch(BATCH_TRAIN_SIZE)



    # `prefetch` lets the dataset fetch batches in the background while the model

    # is training.

    ds = ds.prefetch(buffer_size=AUTOTUNE)



    return ds
train_dsfinal = prepare_for_training(train_ds)

test_dsfinal = prepare_for_training(test_ds)
def show_batch(image_batch, label_batch):

  plt.figure(figsize=(10,10))

  for n in range(25):

      ax = plt.subplot(5,5,n+1)

      plt.imshow(image_batch[n])

      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())

      plt.axis('off')
image_batch, label_batch = next(iter(train_dsfinal))



show_batch(image_batch.numpy(), label_batch.numpy())
label_batch.shape

print(label_batch)
class CNNModel():

    def __init__(self):

        self.inputs = tf.keras.Input(shape=(224,224,3))

        self.x1= tf.keras.layers.Conv2D(32 , 3, activation='relu')(self.inputs)

        self.x1= tf.keras.layers.Conv2D(64, 3, activation='relu')(self.x1)

        self.x1= tf.keras.layers.MaxPooling2D(2,2)(self.x1)

        

        self.x2= tf.keras.layers.Conv2D(32, 3, activation='relu')(self.x1)

        self.x2= tf.keras.layers.Conv2D(64, 3, activation='relu')(self.x2)

        self.x2= tf.keras.layers.MaxPooling2D(3,3)(self.x2)

        

        self.x3= tf.keras.layers.Conv2D(32, 3, activation='relu')(self.x2)

        self.x3= tf.keras.layers.MaxPooling2D(2,2)(self.x3)

        self.x = tf.keras.layers.Dropout(0.2)(self.x3)

        

        self.output = tf.keras.layers.Flatten()(self.x)

        self.output = tf.keras.layers.Dense(224, activation='relu')(self.output)

        self.output = tf.keras.layers.Dense(6, activation='softmax')(self.output) 



        self.model = tf.keras.Model(self.inputs, self.output)

        

        



    def compile_cnn(self):

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

        

    def fit(self, dataset, n_epochs):

        self.model.fit(

            dataset,

            steps_per_epoch=STEPS_PER_EPOCH,

            epochs=n_epochs,

            validation_data=test_dsfinal,

            validation_steps=200

        )



# Create an instance of the model

model = CNNModel()
model.compile_cnn()
history = model.fit(dataset = train_dsfinal, n_epochs = EPOCHS)
acc = model.model.history.history['accuracy']

val_acc = model.model.history.history['val_accuracy']



loss = model.model.history.history['loss']

val_loss = model.model.history.history['val_loss']



epochs_range = range(EPOCHS)



plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
model.model.evaluate(test_dsfinal, verbose=2, steps=64)
model.model.save("model2.h5")
!pip install tensorflowjs
!tensorflowjs_converter --input_format keras model2.h5 tfjs/model