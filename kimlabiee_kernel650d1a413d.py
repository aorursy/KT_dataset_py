import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from PIL import Image

import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()
df = pd.read_csv('../input/train.csv')

# df.head(1)
train_label = np.array([_ for _ in df['has_cactus']])



train_data = []

for filename in '../input/train/train/' + df['id']:

    train_data.append(np.array(Image.open(filename)))

train_data = np.array(train_data)
my_act = tf.keras.layers.LeakyReLU(alpha=0.3)



model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=my_act))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, activation=my_act))

model.add(tf.keras.layers.Dense(32, activation=my_act))

model.add(tf.keras.layers.Dense(16, activation=my_act))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='Adam', loss=['binary_crossentropy'], metrics=['accuracy'])
exit_con = False

while(not exit_con):

    my_model = model.fit(x=train_data, y=train_label, batch_size=32, epochs=1)

    exit_con = my_model.history['acc'][-1]==1.0

#     exit_con = my_model.history['loss'][-1] < 0.005
#plt.plot(my_model.history['loss'])

#my_model.history['acc'][-1]
df_test = pd.read_csv('../input/sample_submission.csv')



test_data = []

for f in '../input/test/test/' + df_test['id']:

    test_data.append(np.array(Image.open(f)))

test_data = np.array(test_data)
df_test['has_cactus'] = model.predict_classes(test_data)
df_test.to_csv("output.csv", index=False)