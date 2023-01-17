# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization, Flatten
from keras.models import Sequential
# from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
batch1 = unpickle("../input/cifar-10-batches-py/data_batch_1")
batch2 = unpickle("../input/cifar-10-batches-py/data_batch_2")
batch3 = unpickle("../input/cifar-10-batches-py/data_batch_3")
batch4 = unpickle("../input/cifar-10-batches-py/data_batch_4")
batch5 = unpickle("../input/cifar-10-batches-py/data_batch_5")
test_batch = unpickle("../input/cifar-10-batches-py/test_batch")
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images

def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls
"""
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """
 # Pre-allocate the arrays for the images and class-numbers for efficiency.
images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
cls = np.zeros(shape=[_num_images_train], dtype=int)
    # Begin-index for the current batch.
begin = 0
    # For each data-file.
for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
    images_batch, cls_batch = _load_data(filename="../input/cifar-10-batches-py/data_batch_" + str(i + 1))

        # Number of images in this batch.
    num_images = len(images_batch)

        # End-index for the current batch.
    end = begin + num_images

        # Store the images into the array.
    images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
    cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
    begin = end

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]
images_train=images
cls_train=cls
labels_train=one_hot_encoded(class_numbers=cls, num_classes=num_classes)
def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="../input/cifar-10-batches-py/test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)
images_test, cls_test, labels_test=load_test_data()
X_test=images_train[:5000,:,:,:]
y_onehot_test=labels_train[:5000,:]
X_train=images_train[5000:,:,:,:]
y_onehot_train=labels_train[5000:,:]

width = X_train[0].shape[0]
height = X_train[0].shape[1]
chans = X_train[0].shape[2]

def decode(one_hot_array):
    normal_array = []
    for i in range(len(one_hot_array)):
        normal_array.append(np.argmax(one_hot_array[i]))
    return normal_array

y_test = decode(y_onehot_test)
y_train = decode(y_onehot_train)
def prep_label(y):
    for i in range(len(y)):
        if y[i] == 0 or y[i] == 1 or y[i] == 8 or y[i] == 9:
            y[i] = 1#1 for vehicle
        else:
            y[i] = 0#0 for animal
            
prep_label(y_train)
prep_label(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_class = y_train.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
def visualize(X, y):
    for i in range(3):
        plt.subplot(331+i)
        plt.imshow(X[i])
        plt.title(y[i])
        
visualize(X_train, y_train)
weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False,
                  pooling='avg',
                  weights=weights_path,
                  input_shape=(width, height, chans)))
model.add(Dense(units=num_class, activation='softmax'))
model.layers[0].trainable = False

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                         horizontal_flip=True,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.3,
                         rotation_range=10,
                         zoom_range=0.1)
train_gen = gen.flow(X_train, y_train, batch_size=100)
val_gen = gen.flow(X_val, y_val, batch_size=100)

model.fit_generator(generator=train_gen,
                    epochs=3,
                    steps_per_epoch=405,
                    validation_data=val_gen,
                    validation_steps=45)
preds = model.predict_classes(X_test, verbose=0)
visualize(X_test, preds)