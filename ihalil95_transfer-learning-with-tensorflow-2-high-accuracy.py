!pip install tensorflow-gpu==2.0.0-beta1
import pathlib

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

print(tf.__version__)
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
train_dir = pathlib.Path('../input/10-monkey-species/training/training')

validation_dir = pathlib.Path('../input/10-monkey-species/validation/validation')
for i, j in zip(train_dir.iterdir(), validation_dir.iterdir()):

    print(i, '  |  ', j)
df = pd.read_csv('../input/10-monkey-species/monkey_labels.txt')

df
train_total = list(train_dir.glob('*/*'))

train_total = len([str(path) for path in train_total])

validation_total = list(validation_dir.glob('*/*'))

validation_total = len([str(path) for path in validation_total])

print(f'{train_total} training example, {validation_total} validation example')
train_image_genarator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,

                                                                     horizontal_flip=True,

                                                                     zoom_range=0.5,

                                                                     rotation_range=45)

validation_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

train_data_gen = train_image_genarator.flow_from_directory(directory=train_dir,

                                                           target_size=(299, 299),

                                                           shuffle=True,

                                                           batch_size=64,

                                                           class_mode='categorical')

validation_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,

                                                                     target_size=(299, 299),

                                                                     batch_size=32,

                                                                     class_mode='categorical')
image, label = next(train_data_gen)

classes = ["alouatta_palliata, mantled_howler", "erythrocebus_patas, patas_monkey", "cacajao_calvus, bald_uakari",

           "macaca_fuscata, japanese_macaque", "cebuella_pygmea, pygmy_marmoset", "cebus_capucinus, white_headed_capuchin",

           "mico_argentatus, silvery_marmoset", "saimiri_sciureus, common_squirrel_monkey",

           "aotus_nigriceps, black_headed_night_monkey", "trachypithecus_johnii, nilgiri_langur"]



plt.figure(figsize=(15, 15))

for i in range(18):

    plt.subplot(6, 3, i+1)

    plt.xticks([])

    plt.yticks([])

    index = np.where(label[i] == 1)[0][0]

    plt.xlabel(classes[index])

    plt.imshow(image[i])

plt.tight_layout()

plt.show()
vimage, vlabel = next(validation_data_gen)

plt.figure(figsize=(15, 15))

for i in range(9):

    plt.subplot(3, 3, i+1)

    plt.xticks([])

    plt.yticks([])

    index = np.where(vlabel[i] == 1)[0][0]

    plt.xlabel(classes[index])

    plt.imshow(vimage[i])

plt.tight_layout()

plt.show()
base_model = keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(299,299,3))

base_model.trainable = False

model = keras.Sequential([

    base_model,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.summary()
keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)
model.compile(keras.optimizers.Adam(9e-5), 'categorical_crossentropy', ['accuracy'])

history = model.fit_generator(train_data_gen,

                              train_total//64,

                              5,

                              validation_data=validation_data_gen,

                              validation_steps=validation_total//32)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = list(range(5))



plt.style.use('seaborn-talk')

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--')

plt.legend(loc='center')

plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle='--')

plt.legend(loc='center')

plt.title('Training and Validation Loss')



plt.tight_layout()

plt.show()