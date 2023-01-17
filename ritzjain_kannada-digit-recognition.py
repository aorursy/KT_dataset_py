# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

from tensorflow.keras import backend as K

import tensorflow as tf
dig_mnist = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

test_Y = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
train_X = np.float64(train_data.loc[:, train_data.columns != 'label'].to_numpy()) / 255

train_X = train_X.reshape(-1, 28, 28, 1)

train_Y = tf.keras.utils.to_categorical(train_data.label, num_classes = 10)



test_X = np.float64(test_data.loc[:, test_data.columns != 'id'].to_numpy()) / 255

test_X = test_X.reshape(-1, 28, 28, 1)

test_Y = test_Y.merge(test_data, on = 'id').label
def display_image(images, size):

     images = images.squeeze()

     plt.figure(figsize = (6, 6))

     n = len(images)

     plt.figure()

     plt.gca().set_axis_off()

     im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])

     for i in range(size)])

     plt.imshow(im)

     plt.show()
display_image(train_X, 5)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

     featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True, 

    rescale = 1./255

)



datagen.fit(train_X.reshape(-1, 28, 28, 1))
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale = 1./255, 

)
model1 = tf.keras.Sequential([

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), input_shape = [28, 28, 1], activation = 'relu'), 

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'), 

    tf.keras.layers.MaxPool2D(), 

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(), 

#     tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(128, activation = 'relu'), 

    tf.keras.layers.Dense(10, activation = 'softmax')

])

model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model1.fit_generator(datagen.flow(train_X.reshape(-1, 28, 28, 1), train_Y, batch_size=32),

                    steps_per_epoch=len(train_X) / 32, epochs=5)
model1_pred = model1.predict_classes(test_X)
submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
submission.drop(['label'], axis = 'columns', inplace = True)
submission['label'] = model1_pred
submission.to_csv("submission.csv", index = False)
model1_pred