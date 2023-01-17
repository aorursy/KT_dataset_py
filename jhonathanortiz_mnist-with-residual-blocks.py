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
import keras
from keras.utils import np_utils
import tensorflow as tf
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt 
train_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#Convert data frame to numpy array and normalize data

train_data = train_dataset.drop(columns=['label']).values.reshape(-1,28,28,1) / 255
label = train_dataset['label'].values



test_data = test_dataset.values.reshape(-1, 28, 28, 1) / 255

print('Test_data', test_data.shape)

print('Train data', train_data.shape)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
plt.figure()
plt.imshow(train_images[0], cmap='gray')
plt.show()
#Reshape the image from (60000, 28,28) to (60000, 28, 28, 1) this new axis represent the color channel, 
#in this case is 1 because de gray scale image only has one color channel

print('Images after reshape', train_images.shape)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
print('Images before resahpe', train_images.shape) 

print('')

#Normalize data from 0-255 to 0-1
print('images after normalize ', train_images[0].max())
train_images /= 255 
test_images /= 255  
print('images before normalize ', train_images[0].max())
#We have 10 numbers types 0 ... 9
n_classes = 10

train_labels = np_utils.to_categorical(train_labels, n_classes)

test_labels = np_utils.to_categorical(train_labels, n_classes)

train_labels.shape
def resblock(inputs, filters, strides):
    
    y = inputs # Shortcut path
    
    # Main path
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Fit shortcut path dimenstions
    if strides > 1:
        y = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
        )(y)
        y = tf.keras.layers.BatchNormalization()(y)

    # Concatenate paths
    x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.ReLU()(x)

    return x
def model():
    
    inputs = tf.keras.layers.Input(shape=(28,28,1))
    
    x = tf.keras.layers.Conv2D(1, (7,7), 1, 'same', use_bias=False)(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    x = resblock(x, 64,  2)
    x = resblock(x, 64,  1)
    
    
    x = tf.keras.layers.AveragePooling2D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(576, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs, x)
    

    
    
model = model()

tf.keras.utils.plot_model(model, show_shapes=True)
model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

history = model.fit(train_data, label, batch_size=128, epochs=10, verbose=1)


results = model.predict(test_data)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)

print(submission.head())

plt.figure()
plt.imshow(test_data[1].reshape(28,28), cmap='gray')
plt.show()

