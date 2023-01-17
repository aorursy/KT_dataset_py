# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import library tensorflow

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# Define f1 metrics
# Diambil dari https://stackoverflow.com/a/45305384/3134677
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Preprocessing for training data

## default image height and width
IMG_HEIGHT, IMG_WIDTH = 256, 256

## Import train image dataset
train_img_dir = '/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/train/'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_img_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode= 'rgb',
    batch_size=50,
    image_size=(IMG_HEIGHT, IMG_WIDTH))

train_ds = train_ds.repeat(9)
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])
train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
# Create CNN Model

model = models.Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))


## Print model summary
model.summary()
# Compile and fit model

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[f1])
history = model.fit(train_ds, epochs=10, verbose=1)
# Preprocessing for test data

## Import test image dataset
test_img_dir = '/kaggle/input/p1-new-dataset/test'
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
directory=test_img_dir,
color_mode= 'rgb',
shuffle=False,
image_size=(IMG_HEIGHT, IMG_WIDTH))

correct_labels = []
for _, labels in test_ds:
    correct_labels += list(labels.numpy())
result = model.predict(test_ds)
# Translate CNN output to class

classes = []
for i in result:
    idx = np.argmax(i)
    classes.append(idx)
# Measure prediction accuracy

print(f1_score(correct_labels, classes, average='macro'))
# Get test dataset ID

fn_sorted = []
for _, _, filenames in os.walk('/kaggle/input/p1-new-dataset/test'):
    for files in filenames:
        fn_sorted.append(files)
fn_sorted = sorted(fn_sorted)
# Prepare submission

submission_df = pd.DataFrame()
submission_df['id'] = fn_sorted
submission_df['label'] = classes
submission_df.to_csv('submission.csv', index=False)