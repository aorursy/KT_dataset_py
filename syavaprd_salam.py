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
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
from math import sin, cos, pi
from tqdm.notebook import tqdm

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from os import listdir
from os.path import join, dirname, abspath
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from os import listdir
from os.path import join, dirname, abspath
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from json import dumps, load
from numpy import array
from os import environ
from os.path import join
from sys import argv
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
from math import sin, cos, pi
import cv2
from tqdm.notebook import tqdm

from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam
path = '../input/00_test_img_input/train/images'
gt = '../input/00_test_img_input/train/gt.csv'
from PIL import Image
jpgfile = Image.open("../input/00_test_img_input/train/images/00020.jpg")
keypoints = res.get('00020.jpg')
jpgfile
fig, axis = plt.subplots()
plot_sample(jpgfile, keypoints, axis, 'sad')
def plot_sample(image, keypoint, axis, title):
    axis.imshow(image)
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)
jpgfile
from skimage.color import rgb2gray
rgb2gray(np.asarray(jpgfile))
fig, axis = plt.subplots()
plot_sample(rgb2gray(np.asarray(jpgfile)), keypoints, axis, 'asd')
def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res
res = read_csv(gt)
res

def convert_y_to_fit(y, x_sizes):
    y_new = np.empty(y.shape)
    y[y < 0] = 0
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] / x_sizes[i, 1] * 96.
        y_new[i, 1:: 2] = y[i, 1:: 2] / x_sizes[i, 0] * 96.

    return y_new

def convert_y_back(y, x_sizes):
    y_new = np.empty(y.shape)
    y[y < 0] = 0
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] * x_sizes[i, 1] / 96.
        y_new[i, 1:: 2] = y[i, 1:: 2] * x_sizes[i, 0] / 96.

    return y_new


file_name = listdir(path)
file_name
fasd = listdir(path)
fasd
def read_images_gt(img_path, size, gt_file=None):
    file_name = listdir(img_path)
    n = len(file_name)
    x = np.empty((n, size, size))
    sizes = np.empty((n, 2))
    y = None
    if gt_file is not None:
        y = np.empty((len(gt_file), 28), dtype=int)

    for i in range(n):
        img = imread(join(img_path, file_name[i]))
        if len(img.shape) == 3:
            img = rgb2gray(img)

        x[i] = resize(img, (size, size))
        sizes[i, 0] = img.shape[0]
        sizes[i, 1] = img.shape[1]
        if gt_file is not None:
            y[i] = gt_file.get(file_name[i])

    x = normalize_x(x)
    x = x.reshape((x.shape[0], size, size, 1))

    if gt_file is not None:
        y = convert_y_to_fit(y, sizes)

    return x, y, sizes
file_name = listdir(path)
img = imread(join(path, file_name[3]))
#img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
y = res.get(file_name[3])
fig, axis = plt.subplots()
plot_sample(img, y, axis, "Sample image & keypoints")
x, y, sizes = read_images_gt(path, 96, res)
len(file_name)
def plot_sample(image, keypoint, axis, title):
    image = image.reshape(96,96)
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)

fig, axis = plt.subplots()
plot_sample(x[4517], y[4517], axis, "Sample image & keypoints")
train_x, train_y = x, y 
def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints
def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints

flipped_train_images, flipped_train_keypoints = left_right_flip(x, y)
print("Shape of flipped_train_images: {}".format(np.shape(flipped_train_images)))
print("Shape of flipped_train_keypoints: {}".format(np.shape(flipped_train_keypoints)))
train_x = np.concatenate((train_x, flipped_train_images))
train_y = np.concatenate((train_y, flipped_train_keypoints))
fig, axis = plt.subplots()
plot_sample(flipped_train_images[3], flipped_train_keypoints[3], axis, "Horizontally Flipped") 
print(train_y.shape)
print(train_x.shape)
rotation_angles = [7, 14]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)
pixel_shifts = [12] 
sample_image_index = 3
def rotate_augmentation(images, keypoints):
    rotated_images = []
    rotated_keypoints = []
    print("Augmenting for angles (in degrees): ")
    for angle in rotation_angles:    # Rotation augmentation for a list of angle values
        for angle in [angle,-angle]:
            print(f'{angle}', end='  ')
            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)
            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
            # For train_images
            for image in images:
                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                rotated_images.append(rotated_image)
            # For train_keypoints
            for keypoint in keypoints:
                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension
                for idx in range(0,len(rotated_keypoint),2):
                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                rotated_keypoint += 48.   # Add the earlier subtracted value
                rotated_keypoints.append(rotated_keypoint)
            
    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints
def rotate_augmentation(images, keypoints):
    rotated_images = []
    rotated_keypoints = []
    print("Augmenting for angles (in degrees): ")
    for angle in rotation_angles:    # Rotation augmentation for a list of angle values
        for angle in [angle,-angle]:
            print(f'{angle}', end='  ')
            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)
            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
            # For train_images
            for image in images:
                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                rotated_images.append(rotated_image)
            # For train_keypoints
            for keypoint in keypoints:
                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension
                for idx in range(0,len(rotated_keypoint),2):
                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                rotated_keypoint += 48.   # Add the earlier subtracted value
                rotated_keypoints.append(rotated_keypoint)
            
    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints


rotated_train_images, rotated_train_keypoints = rotate_augmentation(x, y)
print("\nShape of rotated_train_images: {}".format(np.shape(rotated_train_images)))
print("Shape of rotated_train_keypoints: {}\n".format(np.shape(rotated_train_keypoints)))
train_x = np.concatenate((train_x, rotated_train_images))
train_y = np.concatenate((train_y, rotated_train_keypoints))
fig, axis = plt.subplots()
plot_sample(rotated_train_images[sample_image_index], rotated_train_keypoints[sample_image_index], axis, "Rotation Augmentation")
print(train_y.shape)
print(train_x.shape)
def alter_brightness(images, keypoints): 
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return altered_brightness_images, np.concatenate((keypoints, keypoints))
def alter_brightness(images, keypoints):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return altered_brightness_images, np.concatenate((keypoints, keypoints))


altered_brightness_train_images, altered_brightness_train_keypoints = alter_brightness(x, y)
print(f"Shape of altered_brightness_train_images: {np.shape(altered_brightness_train_images)}")
print(f"Shape of altered_brightness_train_keypoints: {np.shape(altered_brightness_train_keypoints)}")
train_x = np.concatenate((train_x, altered_brightness_train_images))
train_y = np.concatenate((train_y, altered_brightness_train_keypoints))
fig, axis = plt.subplots()
plot_sample(altered_brightness_train_images[sample_image_index], altered_brightness_train_keypoints[sample_image_index], axis, "Increased Brightness") 
fig, axis = plt.subplots()
plot_sample(altered_brightness_train_images[len(altered_brightness_train_images)//2+sample_image_index], altered_brightness_train_keypoints[len(altered_brightness_train_images)//2+sample_image_index], axis, "Decreased Brightness") 
print(train_y.shape)
print(train_x.shape)
def build_model(image_size, output_size):
    model = Sequential()

    # Input dimensions: (None, 96, 96, 1)
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=image_size))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    # Input dimensions: (None, 96, 96, 32)
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Input dimensions: (None, 48, 48, 32)
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    # Input dimensions: (None, 48, 48, 64)
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Input dimensions: (None, 24, 24, 64)
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    # Input dimensions: (None, 24, 24, 96)
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Input dimensions: (None, 12, 12, 96)
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    # Input dimensions: (None, 12, 12, 128)
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Input dimensions: (None, 6, 6, 128)
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    # Input dimensions: (None, 6, 6, 256)
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Input dimensions: (None, 3, 3, 256)
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    # Input dimensions: (None, 3, 3, 512)
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    # Input dimensions: (None, 3, 3, 512)
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(28))
    model.summary()
    return model
NUM_EPOCHS = 75
BATCH_SIZE = 128
%%time

# Define necessary callbacks
checkpointer = ModelCheckpoint(filepath = 'best_model.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])

# Train the model
history = model.fit(train_x, train_y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.05, callbacks=[checkpointer])

features = []
k = 5
for i in range(k):
    features.append((x[i * 6000 // k: (i + 1) * 6000 // k], y[i * 6000 // k: (i + 1) * 6000 // k], sizes[i * 6000 // k: (i + 1) * 6000 // k]))
features[2][1].shape
def compute_metric(detected, values, img_shapes):
    res = 0.0
    for i in range(len(values)):
        n_rows, n_cols = img_shapes[i]
        coords = detected[i]
        diff = (coords - values[i])
        diff[::2] /= n_cols
        diff[1::2] /= n_rows
        diff *= 100
        res += (diff ** 2).mean()
    return res / len(values)
res_s = []
for i in range(1, 2):
    train_feat = []
    train_labels = []
    test = features[i]
    for j in range(k):
        if j == i:
            continue
        train_feat.append(features[i][0])
        train_labels.append(features[i][1])
    train_feat = np.concatenate(train_feat)
    train_labels = np.concatenate(train_labels)
    x_tmp = train_feat
    y_tmp = train_labels
    model = get_model()
    flipped_train_images, flipped_train_keypoints = left_right_flip(x_tmp, y_tmp)
    train_feat = np.concatenate((train_feat, flipped_train_images))
    train_labels = np.concatenate((train_labels, flipped_train_keypoints))
    
    rotated_train_images, rotated_train_keypoints = rotate_augmentation(x_tmp, y_tmp)
    train_feat = np.concatenate((train_feat, rotated_train_images))
    train_labels = np.concatenate((train_labels, rotated_train_keypoints))
    
    altered_brightness_train_images, altered_brightness_train_keypoints = alter_brightness(x_tmp, y_tmp)
    train_feat = np.concatenate((train_feat, altered_brightness_train_images))
    train_labels = np.concatenate((train_labels, altered_brightness_train_keypoints))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])

    # Train the model
    model.fit(train_feat, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    pred_y = model.predict(test[0])
    pred_y = convert_y_back(pred_y, test[2])
    new_test = convert_y_back(test[1], test[2])
    res_s.append(compute_metric(pred_y, new_test, test[2]))
    print(res_s[len(res_s) - 1])
    
def build_model(image_size):
    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=(image_size, image_size, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(28))

    return model
path = '../input/00_test_img_input/train/images'
gt = '../input/00_test_img_input/train/gt.csv'
def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res
res = read_csv(gt)
def normalize_x(x):
    average = np.mean(x, axis=0)
    for i in range(x.shape[0]):
        x[i] -= average

    d1 = np.abs(x.max(axis=0))
    d2 = np.abs(x.min(axis=0))
    d = np.maximum(d1, d2)
    for i in range(x.shape[0]):
        x[i] = (x[i] + d) / (2 * d)

    return x

def read_images_gt(img_path, size, gt_file=None):
    file_name = listdir(img_path)
    n = len(file_name)
    x = np.empty((n, size, size))
    sizes = np.empty((n, 2))
    y = None
    if gt_file is not None:
        y = np.empty((len(gt_file), 28), dtype=int)
    images = []
    for i in range(n):
        #print(file_name[i])
        img = imread(join(img_path, file_name[i]))
        if len(img.shape) == 3:
            img = rgb2gray(img)
        #images.append(img)
        sizes[i, 0] = img.shape[0]
        sizes[i, 1] = img.shape[1]
        x[i] = resize(img, (size, size))
        if gt_file is not None:
            y[i] = gt_file.get(file_name[i])

    x = normalize_x(x)
    x = x.reshape((x.shape[0], size, size, 1))

    if gt_file is not None:
        new_y = convert_y_to_fit(y, sizes)

    return x, new_y, sizes, y
scale = 0.75
x, y, sizes, real_y = read_images_gt(path, 96, res)

train_x = x[:int(len(x) * scale)]
train_y = y[:int(len(y) * scale)]
sizes_train = sizes[:int(len(sizes) * scale)]
real_train = real_y[:int(len(real_y) * scale)]
print(train_x.shape)
print(train_y.shape)
print(sizes_train.shape)
print(real_train.shape)
test_x = x[int(len(x) * scale):]
test_sizes = sizes[int(len(sizes) * scale):]
test_y = real_y[int(len(real_y) * scale):]
test_nonreal_y = y[int(len(y) * scale):]
print(test_x.shape)
print(test_sizes.shape)
print(test_y.shape)
tmp_x = train_x
tmp_y = train_y
tmp_sizes = sizes_train
def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints

flipped_train_images, flipped_train_keypoints = left_right_flip(tmp_x, tmp_y)
print("Shape of flipped_train_images: {}".format(np.shape(flipped_train_images)))
print("Shape of flipped_train_keypoints: {}".format(np.shape(flipped_train_keypoints)))
train_x = np.concatenate((train_x, flipped_train_images))
train_y = np.concatenate((train_y, flipped_train_keypoints))
sizes_train = np.concatenate((sizes_train,tmp_sizes))
fig, axis = plt.subplots()
plot_sample(flipped_train_images[3], flipped_train_keypoints[3], axis, "Horizontally Flipped") 
print(train_x.shape)
print(train_y.shape)
rotation_angles = [12]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)
pixel_shifts = [12] 
sample_image_index = 3
def rotate_augmentation(images, keypoints, sizes):
    rotated_images = []
    rotated_keypoints = []
    new_sizes = []
    cnt = 0
    print("Augmenting for angles (in degrees): ")
    for angle in rotation_angles:    # Rotation augmentation for a list of angle values
        for angle in [angle,-angle]:
            cnt = 0
            print(f'{angle}', end='  ')
            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)
            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
            # For train_images
            for image in images:
                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                rotated_images.append(rotated_image)
                new_sizes.append(sizes[cnt])
                cnt = cnt + 1
            # For train_keypoints
            for keypoint in keypoints:
                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension
                for idx in range(0,len(rotated_keypoint),2):
                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                rotated_keypoint += 48.   # Add the earlier subtracted value
                rotated_keypoints.append(rotated_keypoint)
                
            
    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints, new_sizes


rotated_train_images, rotated_train_keypoints, rotation_sizes = rotate_augmentation(tmp_x, tmp_y, tmp_sizes)
print("\nShape of rotated_train_images: {}".format(np.shape(rotated_train_images)))
print("Shape of rotated_train_keypoints: {}\n".format(np.shape(rotated_train_keypoints)))
train_x = np.concatenate((train_x, rotated_train_images))
train_y = np.concatenate((train_y, rotated_train_keypoints))
sizes_train = np.concatenate((sizes_train, rotation_sizes))
fig, axis = plt.subplots()
plot_sample(rotated_train_images[sample_image_index], rotated_train_keypoints[sample_image_index], axis, "Rotation Augmentation")
print(train_x.shape)
print(train_y.shape)
print(sizes_train.shape)
def alter_brightness(images, keypoints):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return altered_brightness_images, np.concatenate((keypoints, keypoints))


altered_brightness_train_images, altered_brightness_train_keypoints = alter_brightness(tmp_x, tmp_y)
print(f"Shape of altered_brightness_train_images: {np.shape(altered_brightness_train_images)}")
print(f"Shape of altered_brightness_train_keypoints: {np.shape(altered_brightness_train_keypoints)}")
train_x = np.concatenate((train_x, altered_brightness_train_images))
train_y = np.concatenate((train_y, altered_brightness_train_keypoints))
sizes_train = np.concatenate((sizes_train, tmp_sizes))
sizes_train = np.concatenate((sizes_train, tmp_sizes))
fig, axis = plt.subplots()
plot_sample(altered_brightness_train_images[sample_image_index], altered_brightness_train_keypoints[sample_image_index], axis, "Increased Brightness") 
fig, axis = plt.subplots()
plot_sample(altered_brightness_train_images[len(altered_brightness_train_images)//2+sample_image_index], altered_brightness_train_keypoints[len(altered_brightness_train_images)//2+sample_image_index], axis, "Decreased Brightness") 
print(train_x.shape)
print(train_y.shape)
print(sizes_train.shape)
def shift_images(images, keypoints, sizes):
    shifted_images = []
    shifted_keypoints = []
    new_sizes = []
    for shift in pixel_shifts:    # Augmenting over several pixel shift values
        for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
            M = np.float32([[1,0,shift_x],[0,1,shift_y]])
            for image, keypoint, size in zip(images, keypoints, sizes):
                shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
                if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):
                    shifted_images.append(shifted_image.reshape(96,96,1))
                    shifted_keypoints.append(shifted_keypoint)
                    new_sizes.append(size)
    shifted_keypoints = np.clip(shifted_keypoints,0.0,96.0)
    return shifted_images, shifted_keypoints, new_sizes

shifted_train_images, shifted_train_keypoints, shifted_sizes = shift_images(tmp_x, tmp_y, tmp_sizes)
print(f"Shape of shifted_train_images: {np.shape(shifted_train_images)}")
print(f"Shape of shifted_train_keypoints: {np.shape(shifted_train_keypoints)}")
train_x = np.concatenate((train_x, shifted_train_images))
train_y = np.concatenate((train_y, shifted_train_keypoints))
sizes_train = np.concatenate((sizes_train, shifted_sizes))
fig, axis = plt.subplots()
plot_sample(shifted_train_images[sample_image_index], shifted_train_keypoints[sample_image_index], axis, "Shift Augmentation")
print(train_x.shape)
print(train_y.shape)
print(sizes_train.shape)
train_y_sh = convert_y_back(train_y, sizes_train)
model = build_model()
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])

# Train the model
history = model.fit(train_x, train_y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
image_size = 96
#train_x, train_y, _ = read_images_gt(train_img_path, image_size, train_gt)

start = 0.1
stop = 0.001
n_epoch = 100
learning_rate = np.linspace(start, stop, n_epoch)

model = build_model(image_size)
sgd = SGD(lr=start, momentum=0.90, nesterov=True)
model.compile(loss='mse', optimizer=sgd)
change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
early_stop = EarlyStopping(patience=5)
history = model.fit(train_x[:3000], train_y[:3000], epochs=n_epoch, callbacks=[change_lr, early_stop])
test_preds = model.predict(test_x)


def plot_sample(image, keypoint, axis, title):
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)

i = 170
fig, axis = plt.subplots()
plot_sample(test_x[i], test_preds[i], axis, "salan")
i = 171
fig, axis = plt.subplots()
plot_sample(test_x[i], test_preds[i], axis, "salan")
test_sizes[6]
pred_y[3]
pred_y = convert_y_back(test_preds, test_sizes)
test_preds[170]
test_nonreal_y[170]
i = 173
n_rows,n_cols = test_sizes[i]
new_cords = convert_y_back(test_nonreal_y, test_sizes)
coords = new_cords[i]
diff = (coords - test_y[i])
diff[::2] /= n_cols
diff[1::2] /= n_rows
diff *= 100
print((diff ** 2).mean())
def compute_metric(detected, values, img_shapes):
        res = 0.0
        for i in range(len(values)):
            n_rows,n_cols = img_shapes[i]
            coords = detected[i]
            diff = (coords - values[i])
            diff[::2] /= n_cols
            diff[1::2] /= n_rows
            diff *= 100
            res += (diff ** 2).mean()
        return res / len(values)
print(compute_metric(pred_y, test_y, test_sizes))
test_sizes[3]
path = '../input/00_test_img_input/train/images'
gt = '../input/00_test_img_input/train/gt.csv'
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from os import listdir
from os.path import join, dirname, abspath
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from os import listdir
from os.path import join, dirname, abspath
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def build_model(image_size):
    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=(image_size, image_size, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(28))

    return model


def train_detector(train_x, train_y, fast_train=False):
    image_size = 96
    #train_x, train_y, _ = read_images_gt(train_img_path, image_size, train_gt)

    start = 0.1
    stop = 0.001
    n_epoch = 100
    learning_rate = np.linspace(start, stop, n_epoch)

    model = build_model(image_size)
    sgd = SGD(lr=start, momentum=0.90, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    if fast_train:
        model.fit(train_x, train_y, batch_size=200, epochs=1, verbose=0)

    else:
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(patience=5)
        generator = ImageDataGenerator()
        history = model.fit_generator(generator.flow(train_x, train_y), steps_per_epoch=800, epochs=n_epoch, validation_data=(train_x, train_y), callbacks=[change_lr, early_stop])
        #model.save('facepoints_model.hdf5')
        #plot_loss(history.history['loss'], history.history['val_loss'])

    return model


def detect(model, test_x, sizes):
    image_size = 96
    #test_x, _, sizes = read_images_gt(test_img_path, image_size)

    pred_y = model.predict(test_x)
    #pred_y = convert_y_after_detect(pred_y, sizes)

    return pred_y


def read_images_gt(img_path, size, gt_file=None):
    file_name = listdir(img_path)
    n = len(file_name)
    x = np.empty((n, size, size))
    sizes = np.empty((n, 2))
    y = None
    if gt_file is not None:
        y = np.empty((len(gt_file), 28), dtype=int)
    images = []
    for i in range(n):
        #print(file_name[i])
        img = imread(join(img_path, file_name[i]))
        if len(img.shape) == 3:
            img = rgb2gray(img)
        #images.append(img)
        sizes[i, 0] = img.shape[0]
        sizes[i, 1] = img.shape[1]
        x[i] = resize(img, (size, size))
        if gt_file is not None:
            y[i] = gt_file.get(file_name[i])

    x = normalize_x(x)
    x = x.reshape((x.shape[0], size, size, 1))

    if gt_file is not None:
        new_y = convert_y_to_fit(y, sizes)

    return x, new_y, sizes, y

def normalize_x(x):
    average = np.mean(x, axis=0)
    for i in range(x.shape[0]):
        x[i] -= average

    d1 = np.abs(x.max(axis=0))
    d2 = np.abs(x.min(axis=0))
    d = np.maximum(d1, d2)
    for i in range(x.shape[0]):
        x[i] = (x[i] + d) / (2 * d)

    return x


def convert_y_to_fit(y, x_sizes):
    y_new = np.empty(y.shape)
    y[y < 0] = 0
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] / x_sizes[i, 1]
        y_new[i, 1:: 2] = y[i, 1:: 2] / x_sizes[i, 0]
    y_new *= 2
    y_new -= 1

    return y_new

def convert_y_after_detect(y, x_sizes):
    y_new = np.empty(y.shape)
    y += 1
    y /= 2
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] * x_sizes[i, 1]
        y_new[i, 1:: 2] = y[i, 1:: 2] * x_sizes[i, 0]
    y_new = y_new.astype(int)
    return y_new

def plot_loss(loss, val_loss, file_name=join(dirname(abspath('/home/vsevolod/Рабочий стол')), 'history.png')):
    plt.plot(np.arange(len(loss)), np.array(loss), linewidth=3, label='train')
    plt.plot(np.arange(len(val_loss)), np.array(val_loss), linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.savefig(file_name)

class ImageDataGenerator(ImageDataGenerator):
    def next(self):
        X_batch, y_batch = super(ImageDataGenerator, self).next()
        size = 96
        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)

        for i in indices:
            operation = np.random.choice(4, p=np.array((1/8, 3/8, 3/8, 1/8)))

            if operation == 0:
                X_batch[i] = X_batch[i, :, ::-1, :]
                y_batch[i, ::2] = y_batch[i, ::2] * -1

            if operation == 1:
                angle = np.arange(5, 25, 5)[random.randint(0, 4)]
                M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
                X_batch[i, :, :, 0] = cv2.warpAffine(X_batch[i, :, :, 0], M, (size, size))

                tmp = y_batch[i]
                tmp = (tmp + 1) / 2 * size
                tmp = tmp.reshape((tmp.size // 2, 2))
                ones = np.ones(shape=(tmp.shape[0], 1))
                points_ones = np.hstack([tmp, ones])
                tmp = M.dot(points_ones.T).T
                tmp = tmp.reshape(tmp.size)
                tmp = tmp / size * 2 - 1
                tmp[tmp < -1] = -1
                tmp[tmp > 1] = 1
                y_batch[i] = tmp

            if operation == 2:
                angle = np.arange(5, 25, 5)[random.randint(0, 4)]
                M = cv2.getRotationMatrix2D((size // 2, size // 2), -angle, 1.0)
                X_batch[i, :, :, 0] = cv2.warpAffine(X_batch[i, :, :, 0], M, (size, size))

                tmp = y_batch[i]
                tmp = (tmp + 1) / 2 * size
                tmp = tmp.reshape((tmp.size // 2, 2))
                ones = np.ones(shape=(tmp.shape[0], 1))
                points_ones = np.hstack([tmp, ones])
                tmp = M.dot(points_ones.T).T
                tmp = tmp.reshape(tmp.size)
                tmp = tmp / size * 2 - 1
                tmp[tmp < -1] = -1
                tmp[tmp > 1] = 1
                y_batch[i] = tmp

            if operation == 3:
                tmp = y_batch[i]
                tmp = (tmp + 1) / 2 * size
                tmp = tmp.reshape((tmp.size // 2, 2))
                max_x = tmp[0].max()
                max_y = tmp[1].max()
                min_x = tmp[0].min()
                min_y = tmp[1].min()
                if max_x - min_x < size - 1 and max_y - min_y < size - 1:
                    new_size = random.randint(max(max_x - min_x, max_y - min_y), size - 1)
                    x = random.randint(max(0, max_x - new_size), min_x)
                    y = random.randint(max(0, max_y - new_size), min_y)
                    X_batch[i, :, :, 0] = resize(X_batch[i, y: y + new_size, x: x + new_size, 0], (size, size))
                    tmp[0] -= x
                    tmp[1] -= y
                    tmp = tmp.reshape(tmp.size)
                    tmp = tmp / new_size * 2 - 1
                    y_batch[i] = tmp

        return X_batch, y_batch
def train_detector(train_x, train_y, fast_train=False):
    image_size = 96
    #train_x, train_y, _ = read_images_gt(train_img_path, image_size, train_gt)

    start = 0.1
    stop = 0.001
    n_epoch = 100
    learning_rate = np.linspace(start, stop, n_epoch)

    model = build_model(image_size)
    sgd = SGD(lr=start, momentum=0.90, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    if fast_train:
        model.fit(train_x, train_y, batch_size=200, epochs=1, verbose=0)

    else:
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(patience=5)
        generator = ImageDataGenerator()
        history = model.fit_generator(generator.flow(train_x, train_y), steps_per_epoch=500, epochs=n_epoch, validation_data=(train_x, train_y), callbacks=[change_lr, early_stop])
        #model.save('facepoints_model.hdf5')
        #plot_loss(history.history['loss'], history.history['val_loss'])

    return model


def detect(model, test_x, sizes):
    image_size = 96
    #test_x, _, sizes = read_images_gt(test_img_path, image_size)

    pred_y = model.predict(test_x)
    #pred_y = convert_y_after_detect(pred_y, sizes)

    return pred_y
def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res
res = read_csv(gt)
scale = 0.75
x, y, sizes, real_y = read_images_gt(path, 96, res)
train_x = x[:int(len(x) * scale)]
train_y = y[:int(len(y) * scale)]
sizes_train = sizes[:int(len(sizes) * scale)]
real_train = real_y[:int(len(real_y) * scale)]
print(train_x.shape)
print(train_y.shape)
print(sizes_train.shape)
print(real_train.shape)
test_x = x[int(len(x) * scale):]
test_sizes = sizes[int(len(sizes) * scale):]
test_y = real_y[int(len(real_y) * scale):]
test_nonreal_y = y[int(len(y) * scale):]
def convert_y_to_fit(y, x_sizes):
    y_new = np.empty(y.shape)
    y[y < 0] = 0
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] / x_sizes[i, 1]
        y_new[i, 1:: 2] = y[i, 1:: 2] / x_sizes[i, 0]
    y_new *= 2
    y_new -= 1

    return y_new

def convert_y_after_detect(y, x_sizes):
    y_new = np.empty(y.shape)
    y += 1
    y /= 2
    for i in range(y_new.shape[0]):
        y_new[i, :: 2] = y[i, :: 2] * x_sizes[i, 1]
        y_new[i, 1:: 2] = y[i, 1:: 2] * x_sizes[i, 0]
    y_new = y_new.astype(int)
    return y_new
train_sh_fit = convert_y_to_fit(train_y_sh, sizes_train)
train_sh_fit
model = train_detector(train_x, train_y)
pred_Y = detect(model, test_x, test_sizes)
last_preds = convert_y_after_detect(pred_Y, test_sizes)
def compute_metric(detected, values, img_shapes):
    res = 0.0
    for i in range(len(values)):
        n_rows, n_cols = img_shapes[i]
        coords = detected[i]
        diff = (coords - values[i])
        diff = diff.astype(float)
        diff[::2] /= n_cols
        diff[1::2] /= n_rows
        diff *= 100
        res += (diff ** 2).mean()
    return res / len(values)
compute_metric(last_preds, test_y, test_sizes)

def alter_brightness(image, keypoints, alpha):
    bright_image = np.clip(image*alpha, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    return bright_image, keypoints
def shift_images(image, keypoints, shift_x, shift_y): 
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
    shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
    if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):
        return (shifted_image, shifted_keypoint)
    return (None, None)


import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from os import listdir
from os.path import join, dirname, abspath
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


import time
import glob
import numpy as np
import skimage.io as skimio
import skimage.transform as skimtr
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout
)
from keras.optimizers import SGD
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint
)


IMG_EDGE_SIZE = 100
IMG_SHAPE = (IMG_EDGE_SIZE, IMG_EDGE_SIZE)

GSCALE = True
INPUT_SHAPE = (*IMG_SHAPE, 1 if GSCALE else 3)  # input layer shape

Y_BIAS = 50  # IMG_EDGE_SIZE / 2
Y_NORM = 10  # IMG_EDGE_SIZE / 2
MIN_ROTATION_ANGLE = 5  # in degrees
MAX_ROTATION_ANGLE = 15  # in degrees


def rotate_img(img, y, alphas):
    # alpha = 2 * MAX_ROTATE_ANGLE * (np.random.rand() - 0.5)
    alpha = np.random.choice(alphas)
    alpha_rad = np.radians(alpha)
    rot_mat = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                        [np.sin(alpha_rad), np.cos(alpha_rad)]])
    bias = img.shape[0] / 2
    return (
        skimtr.rotate(img, alpha),
        (y - bias).reshape(-1, 2).dot(rot_mat).ravel() + bias
    )


def cut_img(img, y):
    h = img.shape[0]
    lt = int(np.ceil(min(np.random.randint(0.05 * h, 0.15 * h), y.min())))
    rb = int(np.ceil(max(np.random.randint(0.85 * h, 0.95 * h), y.max())))
    return img[lt: rb, lt: rb], y - lt


def flip_img(img, y):
    y_ = y.copy()
    y_[::2] = img.shape[1] - y_[::2] - 1
    return (
        img[:, ::-1],
        y_.reshape(-1, 2)[
            [3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 10, 13, 12, 11]
        ].ravel()
    )


def load_data(img_dir, gt, input_shape, output_size=28, test=False, test_k = 1500):
    print("STARTED LOADING DATA")
    _start = time.time()

    N = len(gt) - test_k
    rotations_num = 4
    cut_num = 1
    brightness_num = 4

    start_flipped = N
    start_rotation = start_flipped + N
    start_cut = start_rotation + rotations_num * N
    start_brightness = start_cut + cut_num * N
    if not test:
        N = 2 * N + rotations_num * N + cut_num * N + brightness_num * N # one N for flipped imgs
    #if not test:
        
    X = np.empty((N, *input_shape))
    y = np.empty((N, output_size)) if not test else None
    scales = []
    test_scales = []
    X_test = np.empty((test_k, *input_shape))
    y_test = np.empty((test_k, output_size)) if not test else None
    test_fn = []
    for i, (fn, y_raw) in enumerate(gt.items()):
        if i >= 4500:
            img = skimio.imread(img_dir + '/' + fn, as_gray=GSCALE)
            scale_y = 1.0 * img.shape[0] / input_shape[0]
            scale_x = 1.0 * img.shape[1] / input_shape[1]
            test_scales += [(scale_x, scale_y, fn)] 
            
            X_test[i - 4500] = skimtr.resize(img, input_shape, mode='reflect')

            if not test:
                # Original image
                y[i - 4500][::2] = y_raw[::2] / scale_x
                y[i - 4500][1::2] = y_raw[1::2] / scale_y
            continue
            
        img = skimio.imread(img_dir + '/' + fn, as_gray=GSCALE)
        #print(img)
        #input1 = input()
        scale_y = 1.0 * img.shape[0] / input_shape[0]
        scale_x = 1.0 * img.shape[1] / input_shape[1]
        scales += [(scale_x, scale_y, fn)]

        X[i] = skimtr.resize(img, input_shape, mode='reflect')

        if not test:
            # Original image
            y[i][::2] = y_raw[::2] / scale_x
            y[i][1::2] = y_raw[1::2] / scale_y

            # Flipped image
            X[start_flipped + i], y[start_flipped + i] = flip_img(X[i], y[i])

            # Rotated images
            for r in range(rotations_num):
                indx = start_rotation + rotations_num * i + r
                if r == 0:
                    alphas = list(
                        range(-MAX_ROTATION_ANGLE, -MIN_ROTATION_ANGLE + 1))
                else:
                    alphas = list(
                        range(MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE + 1))
                X[indx], y[indx] = rotate_img(X[i], y[i], alphas)

            # Cutted images
            for c in range(cut_num):
                indx = start_cut + cut_num * i + c
                if y_raw.min() < 0:
                    X[indx], y[indx] = X[i], y[i]
                else:
                    img_c, y_c = cut_img(img, y_raw)
                    scale = 1.0 * img_c.shape[0] / input_shape[0]
                    X[indx] = skimtr.resize(img_c, input_shape, mode='reflect')
                    y[indx] = y_c / scale
            random_range = [(1.1, 1.4), (0.5, 0.8), (1.1, 1.4), (0.5, 0.8)]
            #brightness images
            for c in range(brightness_num):
                indx = start_brightness + brightness_num * i + c
                random_range_c = random_range[c]
                img_c, y_c = alter_brightness(X[i], y[i], random.uniform(random_range_c[0], random_range_c[1]))
                X[indx] = img_c
                y[indx] = y_c

    if not test:
        y = (y - Y_BIAS) / Y_NORM
        
    if not test:
        y_test = (y_test - Y_BIAS) / Y_NORM

    mean = np.mean(X, axis=0)
    std = (np.mean(X ** 2, axis=0) - mean ** 2) ** 0.5
    X = (X - mean) / std
    
    mean = np.mean(X_test, axis=0)
    std = (np.mean(X_test ** 2, axis=0) - mean ** 2) ** 0.5
    X_test = (X_test - mean) / std

    print("FINISHED LOADING DATA:", time.time() - _start)

    return X, y, scales, X_test, y_test, test_scales


def build_model(image_size, output_size):
    model = Sequential()
    #Added
    model.add(Conv2D(16, 3, input_shape=image_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(1000))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(28))
    
    return model


def train_detector(X, y, fast_train=False):
    input_shape = INPUT_SHAPE  # input layer shape
    train_gt = res
    output_size = len(list(train_gt.values())[0])
    if fast_train:
        keys = list(train_gt.keys())[:10]
        train_gt = {key: train_gt[key] for key in keys}

    #X, y, _ = load_data(train_img_dir, train_gt, input_shape, output_size)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

    # Model config.
    epochs = 1 if fast_train else 350
    patience = 50  # stop if err has not been updated patience time
    early_stop = EarlyStopping(patience=patience)

    # SGD config.
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # Model setup
    model = build_model(input_shape, output_size)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    checkpoint_callback = ModelCheckpoint(filepath='mod1el.hdf5',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')

    # Model training
    start_time = time.time()
    print('start_time: {}'.format(time.strftime('%H:%M:%S')))
    model.fit(
        X_tr, y_tr,
        epochs=epochs,
        validation_data=(X_te, y_te),
        callbacks=[early_stop, checkpoint_callback]
    )
    print('end_time: {}, duration(min): {}'.format(time.strftime('%H:%M:%S'),
          (time.time()-start_time) / 60.))

    return model


def detect(model, test_img_dir):
    gt = {}
    for fname in glob.glob1(test_img_dir, '*.jpg'):
        gt[fname] = None

    X, _, scales = load_data(test_img_dir, gt, INPUT_SHAPE, test=True)
    y_pred = model.predict(X) * Y_NORM + Y_BIAS

    y_scaled = {}
    for i in range(len(scales)):
        scale_x, scale_y, fn = scales[i]
        y = y_pred[i]
        y[::2] = y[::2] * scale_x
        y[1::2] = y[1::2] * scale_y
        y_scaled[fn] = y

    return y_scaled
X, y, _, X_test, y_test, test_scales = load_data(path, res, INPUT_SHAPE, 28)
X.shape
print('Done')
for t in _:
    if t[2] in test_scales[2]:
        print('x')
def detect(model, X,scales):
    gt = {}
    for i in scales:
        fname = i[2]
        gt[fname] = None

    #X, _, scales = load_data(test_img_dir, gt, INPUT_SHAPE, test=True)
    y_pred = model.predict(X) * Y_NORM + Y_BIAS

    y_scaled = {}
    for i in range(len(scales)):
        scale_x, scale_y, fn = scales[i]
        y = y_pred[i]
        y[::2] = y[::2] * scale_x
        y[1::2] = y[1::2] * scale_y
        y_scaled[fn] = y

    return y_scaled
model = train_detector(X, y)
train_gt = res
from tensorflow.keras.models import load_model
model = load_model('mod1el.hdf5')

test_y = detect(model, X_test, test_scales)
def compute_metric(detected, gt, img_shapes):
    res = 0.0
    for filename, coords in detected.items():
        n_rows, n_cols = img_shapes[filename]
        diff = (coords - gt[filename]) 
        diff[::2] /= n_cols
        diff[1::2] /= n_rows
        diff *= 100
        res += (diff ** 2).mean()
    return res / len(detected.keys())
def read_img_shapes(gt_dir):
        img_shapes = {}
        with open(join(gt_dir, 'img_shapes.csv')) as fhandle:
            next(fhandle)
            for line in fhandle:
                parts = line.rstrip('\n').split(',')
                filename = parts[0]
                n_rows, n_cols = map(int, parts[1:])
                img_shapes[filename] = (n_rows, n_cols)
        return img_shapes
img_shapes = read_img_shapes('../input/00_test_img_gt')
compute_metric(test_y, res, img_shapes)
enumerate(res.items())

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from os import listdir
from os.path import join, dirname, abspath
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
import glob
import numpy as np
import skimage.io as skimio
import skimage.transform as skimtr
from sklearn.model_selection import train_test_split
from numpy import array

from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout
)
from keras.optimizers import SGD
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint
)

IMG_EDGE_SIZE = 100
IMG_SHAPE = (IMG_EDGE_SIZE, IMG_EDGE_SIZE)

GSCALE = True
INPUT_SHAPE = (*IMG_SHAPE, 1 if GSCALE else 3)  # input layer shape

Y_BIAS = 50  # IMG_EDGE_SIZE / 2
Y_NORM = 10  # IMG_EDGE_SIZE / 2
MIN_ROTATION_ANGLE = 5  # in degrees
MAX_ROTATION_ANGLE = 15  # in degrees


def rotate_img(img, y, alphas):
    # alpha = 2 * MAX_ROTATE_ANGLE * (np.random.rand() - 0.5)
    alpha = np.random.choice(alphas)
    alpha_rad = np.radians(alpha)
    rot_mat = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                        [np.sin(alpha_rad), np.cos(alpha_rad)]])
    bias = img.shape[0] / 2
    return (
        skimtr.rotate(img, alpha),
        (y - bias).reshape(-1, 2).dot(rot_mat).ravel() + bias
    )


def cut_img(img, y):
    h = img.shape[0]
    lt = int(np.ceil(min(np.random.randint(0.05 * h, 0.15 * h), y.min())))
    rb = int(np.ceil(max(np.random.randint(0.85 * h, 0.95 * h), y.max())))
    return img[lt: rb, lt: rb], y - lt


def flip_img(img, y):
    y_ = y.copy()
    y_[::2] = img.shape[1] - y_[::2] - 1
    return (
        img[:, ::-1],
        y_.reshape(-1, 2)[
            [3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 10, 13, 12, 11]
        ].ravel()
    )


def alter_brightness(image, keypoints, alpha):
    bright_image = np.clip(image*alpha, 0.0, 1.0) 
    return bright_image, keypoints

def shift_images(image, keypoints, shift_x, shift_y): 
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
    shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
    if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):
        return (shifted_image, shifted_keypoint)
    return (None, None)

def load_data(img_dir, gt, input_shape, output_size=28, test=False):
    N = len(gt)
    rotations_num = 3
    cut_num = 1
    brightness_num = 3

    start_flipped = N
    start_rotation = start_flipped + N
    start_cut = start_rotation + rotations_num * N
    start_brightness = start_cut + cut_num * N
    if not test:
        N = 2 * N + rotations_num * N + cut_num * N + brightness_num * N 

        
    X = np.empty((N, *input_shape))
    y = np.empty((N, output_size)) if not test else None
    scales = []
    test_fn = []
    for i, (fn, y_raw) in enumerate(gt.items()):
        img = skimio.imread(img_dir + '/' + fn, as_gray=GSCALE)
        #print(img)
        #input1 = input()
        scale_y = 1.0 * img.shape[0] / input_shape[0]
        scale_x = 1.0 * img.shape[1] / input_shape[1]
        scales += [(scale_x, scale_y, fn)]

        X[i] = skimtr.resize(img, input_shape, mode='reflect')

        if not test:
            # Original image
            y[i][::2] = y_raw[::2] / scale_x
            y[i][1::2] = y_raw[1::2] / scale_y

            # Flipped image
            X[start_flipped + i], y[start_flipped + i] = flip_img(X[i], y[i])

            # Rotated images
            for r in range(rotations_num):
                indx = start_rotation + rotations_num * i + r
                if r == 0:
                    alphas = list(
                        range(-MAX_ROTATION_ANGLE, -MIN_ROTATION_ANGLE + 1))
                else:
                    alphas = list(
                        range(MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE + 1))
                X[indx], y[indx] = rotate_img(X[i], y[i], alphas)

            # Cutted images
            for c in range(cut_num):
                indx = start_cut + cut_num * i + c
                if y_raw.min() < 0:
                    X[indx], y[indx] = X[i], y[i]
                else:
                    img_c, y_c = cut_img(img, y_raw)
                    scale = 1.0 * img_c.shape[0] / input_shape[0]
                    X[indx] = skimtr.resize(img_c, input_shape, mode='reflect')
                    y[indx] = y_c / scale
            random_range = [(1.1, 1.4), (0.5, 0.8), (1.1, 1.4), (0.5, 0.8)]
            #brightness images
            for c in range(brightness_num):
                indx = start_brightness + brightness_num * i + c
                random_range_c = random_range[c]
                img_c, y_c = alter_brightness(X[i], y[i], random.uniform(random_range_c[0], random_range_c[1]))
                X[indx] = img_c
                y[indx] = y_c

    if not test:
        y = (y - Y_BIAS) / Y_NORM
        

    mean = np.mean(X, axis=0)
    std = (np.mean(X ** 2, axis=0) - mean ** 2) ** 0.5
    X = (X - mean) / std

    return X, y, scales


def build_model(image_size, output_size):
    model = Sequential()
    #Added
    model.add(Conv2D(16, 3, input_shape=image_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(1000))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(28))
    
    return model


def train_detector(train_gt, train_img_dir, fast_train=False):
    input_shape = INPUT_SHAPE  # input layer shape
    train_gt = res
    output_size = len(list(train_gt.values())[0])
    if fast_train:
        keys = list(train_gt.keys())[:10]
        train_gt = {key: train_gt[key] for key in keys}

    X, y, _ = load_data(train_img_dir, train_gt, input_shape, output_size)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

    # Model config.
    epochs = 1 if fast_train else 350
    patience = 50  # stop if err has not been updated patience time
    early_stop = EarlyStopping(patience=patience)

    # SGD config.
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # Model setup
    model = build_model(input_shape, output_size)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    checkpoint_callback = ModelCheckpoint(filepath='facepoints_model.hdf5',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')

    model.fit(
        X_tr, y_tr,
        epochs=epochs,
        validation_data=(X_te, y_te),
        callbacks=[early_stop, checkpoint_callback]
    )

    return model


def detect(model, test_img_dir):
    gt = {}
    for fname in glob.glob1(test_img_dir, '*.jpg'):
        gt[fname] = None

    X, _, scales = load_data(test_img_dir, gt, INPUT_SHAPE, test=True)
    y_pred = model.predict(X) * Y_NORM + Y_BIAS

    y_scaled = {}
    for i in range(len(scales)):
        scale_x, scale_y, fn = scales[i]
        y = y_pred[i]
        y[::2] = y[::2] * scale_x
        y[1::2] = y[1::2] * scale_y
        y_scaled[fn] = y

    return y_scaled
path = '../input/00_test_img_input/train/images'
gt = '../input/00_test_img_input/train/gt.csv'
def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res
res = read_csv(gt)
train_detector(res, path)
test_path = '../input/00_test_img_input/test/images'
from tensorflow.keras.models import load_model
model = load_model('facepoints_model.hdf5')
preds = detect(model, test_path)
def compute_metric(detected, gt, img_shapes):
    res = 0.0
    for filename, coords in detected.items():
        n_rows, n_cols = img_shapes[filename]
        diff = (coords - gt[filename])
        diff[::2] /= n_cols
        diff[1::2] /= n_rows
        diff *= 100
        res += (diff**2).mean()
    return res/len(detected.keys())
compute_metric(preds, res, img_shapes)
