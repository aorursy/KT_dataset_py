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
import tensorflow as tf

print(tf.__version__)



import pandas as pd

import numpy as np
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
test_df.head()
y_train = train_df.label.values

x_train = train_df.drop(columns=["label"]).values

x_test = test_df.values

x_train[:10]
print(x_train.shape)

print(x_test.shape)

print(max(x_train[1]))
x_train = x_train / 255.0

x_test = x_test / 255.0

model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(784,)),

    # tf.keras.layers.Dense(2048, activation=tf.nn.relu),

    # tf.keras.layers.Dense(1024, activation=tf.nn.relu),

    # tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dense(256, activation=tf.nn.relu),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('acc') > 0.997):

            print("\n Reached 99% accuracy so cancelling training!")

            self.model.stop_training = True





model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30)

classifications = model.predict(x_test)
classifications.shape

classifications

np.argmax(classifications, axis=1)[0]
def write_submissions(file_name, imageId, predictions):

    

    output = pd.DataFrame({

        'ImageId': imageId, 'Label': predictions

    })

    output.to_csv(file_name, index=False)

    

    

write_submissions('submission_1.csv', pd.Series(range(1,28001)), np.argmax(classifications, axis=1))
y_train = train_df.label.values

x_train = train_df.drop(columns=["label"]).values

x_test = test_df.values
print(x_train.shape)

print(x_test.shape)
x_train = x_train.reshape(42000, 28, 28, 1)

x_test = x_test.reshape(28000, 28, 28, 1)



x_train = x_train / 255.0

x_test = x_test / 255.0
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    # tf.keras.layers.Dense(2048, activation='relu'),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(256, activation='relu'),

    # tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=20)
classifications = model.predict(x_test)

write_submissions('submission_2.csv', pd.Series(range(1,28001)), np.argmax(classifications, axis=1))
print(y_train[:100])

x_train.shape
import matplotlib.pyplot as plt

f, axarr = plt.subplots(3, 4)

FIRST_IMAGE = 0

SECOND_IMAGE = 7

THIRD_IMAGE = 8

CONVOLUTION_NUMBER = 1



from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers]

activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)



for x in range(0, 4):

    f1 = activation_model.predict(x_train[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]

    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')

    axarr[0, x].grid(False)

    f2 = activation_model.predict(x_train[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]

    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')

    axarr[1, x].grid(False)

    f3 = activation_model.predict(x_train[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]

    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')

    axarr[2, x].grid(False)

    