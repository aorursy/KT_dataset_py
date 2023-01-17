# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras import layers

import numpy as np

import keras as keras

from tensorflow.python.keras.models import Sequential

import matplotlib.pyplot as plt

import PIL



!pip install tf-nightly

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

# training and validation data

batch_size = 32

img_height = 150

img_width = 150

train_data_dir = "../input/intel-image-classification/seg_train/seg_train/"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    data_dir,

    seed=123,

    image_size=(img_height, img_width),

    batch_size=batch_size)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

  for i in range(6):

    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(class_names[labels[i]])

    plt.axis("off")
batch_size = 32

img_height = 150

img_width = 150

test_data_dir = "../input/intel-image-classification/seg_test/seg_test/"

test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    test_data_dir,

    seed=123,

    image_size=(img_height, img_width),

    batch_size=batch_size)
# class names

class_names = train_ds.class_names

print(class_names)



for image_batch, labels_batch in train_ds:

    print(image_batch.shape)

    print(labels_batch.shape)

    break
# feature scaling

normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)



normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_ds))

first_image = image_batch[0]

# Notice the pixels values are now in `[0,1]`.

print(np.min(first_image), np.max(first_image))
# creating the model

num_classes = 6

model = Sequential([

    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(16, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(num_classes)

])
# Compiling the model

model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])



model.summary()
epochs = 10

history = model.fit(

    train_ds,

    validation_data=test_ds,

    epochs=epochs

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

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
model.evaluate(test_ds)
PIL.Image.open("../input/intel-image-classification/seg_pred/seg_pred/10013.jpg")
PIL.Image.open("../input/intel-image-classification/seg_pred/seg_pred/10017.jpg")
PIL.Image.open("../input/intel-image-classification/seg_pred/seg_pred/10021.jpg")
image_paths = ['10013.jpg', '10017.jpg', '10021.jpg']



for image_path in image_paths:

    root_path = "../input/intel-image-classification/seg_pred/seg_pred/"

    path = root_path + image_path

    image = tf.keras.preprocessing.image.load_img(path)

    input_arr = keras.preprocessing.image.img_to_array(image)

    input_arr = np.array([input_arr])  # Convert single image to a batch.

    score = tf.nn.softmax(model.predict(input_arr))

    print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score)))