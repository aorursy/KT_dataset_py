# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split



df_train = pd.read_csv('../input/train.csv')

df_train.head()
plt.figure(figsize=(10, 8))

for i in range(0, 20):

    plt.subplot(4, 5, i+1)

    data = df_train.loc[i]

    img = mpimg.imread('../input/train/train/{}'.format(data.id))

    plt.imshow(img / 255, 'gray')

    plt.title('Cactus' if data.has_cactus else 'No Cactus')

    plt.xticks([])

    plt.yticks([])
filenames = ['../input/train/train/' + fname for fname in df_train['id'].tolist()]

labels = df_train['has_cactus'].tolist()



train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames, labels, train_size=0.9, random_state=42)
train_data = tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))

val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels)))
def convert_image(filename, label=''):

    img = tf.io.read_file(filename)

    img = tf.image.decode_jpeg(img)

    img = (tf.cast(img, tf.float32)/127.5)-1

    img = tf.image.resize(img, (32, 32))

    return img, label
train_data = (train_data.map(convert_image).shuffle(buffer_size=10000).batch(32))

val_data = (val_data.map(convert_image).shuffle(buffer_size=10000).batch(32))
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit(train_data.repeat(), epochs=20, steps_per_epoch=round(len(train_filenames)/32), validation_data=val_data.repeat(), validation_steps=20)
fig = plt.figure(figsize=(18, 6))



plt.subplot2grid((2, 3), (0, 0))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')



plt.subplot2grid((2, 3), (0, 1))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')



plt.show()
import glob

filenames = glob.glob('../input/test/test/*.jpg')



test_data = tf.data.Dataset.from_tensor_slices((tf.constant(filenames)))

test_data = test_data.map(convert_image)

test_data = test_data.batch(32)
predictions = model.predict(test_data, steps=len(filenames))
ids = [fname.split('/')[4] for fname in filenames]

has_cactus = predictions
plt.figure(figsize=(12, 8))

for i in range(0, len(predictions[:25])):

    plt.subplot(5, 5, i+1)

    img = mpimg.imread(filenames[i])

    plt.imshow(img)

    plt.title('Cactus' if predictions[i] >= 0.5 else 'No Cactus')

    plt.xticks([])

    plt.yticks([])
import csv



csv_data = [['id', 'has_cactus']]



for i in range(0, len(ids)):

    csv_data.append([ids[i], has_cactus[i][0]])



with open('submission.csv', 'w') as csv_file:

    writer = csv.writer(csv_file)

    writer.writerows(csv_data)



csv_file.close()
!cat submission.csv