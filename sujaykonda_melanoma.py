# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras import layers

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
trainrecords = []
testrecords = []
for dirname, _, filenames in os.walk("../input/siim-isic-melanoma-classification/tfrecords"):
    for filename in filenames:
        if(filename.startswith("train")):
            trainrecords.append(os.path.join(dirname, filename))
        else:
            testrecords.append(os.path.join(dirname, filename))
print(trainrecords)
def _parse_function(record):
    features = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "image_name": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "target": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    }
    
    parsed_features = tf.io.parse_single_example(record, features)
    
    image_shape = tf.stack([1024, 1024, 3])
    
    image = tf.image.decode_jpeg(parsed_features["image"])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, image_shape)
    image = tf.image.resize(image, (96, 96))
    image = tf.image.rgb_to_grayscale(image)
    mean = tf.math.reduce_mean(image)
    image = image - mean
    image = image/(tf.math.reduce_max(image) - tf.math.reduce_min(image))
    
    
    target = parsed_features["target"]
    target = tf.cast(target, tf.float32)
    
    image_name = parsed_features["image_name"]
    return image, target
rawtrainds = tf.data.TFRecordDataset(trainrecords)
rawtestds = tf.data.TFRecordDataset(testrecords)
trainds = rawtrainds.map(_parse_function)
x_all = []
y_all = []
for x_val, y_val in trainds.as_numpy_iterator():
    x_all.append(x_val)
    y_all.append(y_val)
x_all = np.array(x_all)
y_all = np.array(y_all)
total = 30000
x_train = x_all[:total]
x_test = x_all[total:]
y_train = y_all[:total]
y_test = y_all[total:]
weight_for_0 = (1 / (total - sum(y_train)))*(total)/2.0 
weight_for_1 = (1 / sum(y_train))*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}
model = keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=5, activation="relu", input_shape=(96,96,1)))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=21, batch_size=100, class_weight=class_weight)
model.evaluate(x_test, y_test, batch_size=100)
def _parse_test_function(record):
    features = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "image_name": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    
    parsed_features = tf.io.parse_single_example(record, features)
    
    image_shape = tf.stack([1024, 1024, 3])
    
    image = tf.image.decode_jpeg(parsed_features["image"])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, image_shape)
    image = tf.image.resize(image, (96, 96))
    image = tf.image.rgb_to_grayscale(image)
    mean = tf.math.reduce_mean(image)
    image = image - mean
    image = image/(tf.math.reduce_max(image) - tf.math.reduce_min(image))
    
    image_name = parsed_features["image_name"]
    return image_name, image
testds = rawtrainds.map(_parse_test_function)
sub_test = []
sub_image_name = []
for img_name, img in testds.as_numpy_iterator():
        sub_image_name.append(img_name)
        sub_test.append(img)
sub_test = np.array(sub_test)
sub_image_name = np.array(sub_image_name)
sub_target = model.predict(sub_test)

my_submission = pd.DataFrame({'image_name': sub_image_name, 'target': sub_target.flatten()})
my_submission.to_csv('submission.csv', index=False)
