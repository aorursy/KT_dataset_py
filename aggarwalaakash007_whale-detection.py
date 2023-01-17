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
INPUT_DIR = '../input'
import math
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

from tqdm import tqdm

%matplotlib inline
#to plot images
def plot_images_for_filenames(filenames, labels, rows=4):
    imgs = [plt.imread(f'{INPUT_DIR}/train/{filename}') for filename in filenames]
    
    return plot_images(imgs, labels, rows)
def plot_images(imgs, labels, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i] , cmap = 'gray')
np.random.seed(0)
data_train = pd.read_csv("../input/train.csv")
print(data_train.head())
rand_rows = data_train.sample(frac = 1.0)[:25]
imgs = list(rand_rows['Image'])
print(imgs)
labels = list(rand_rows['Id'])

plot_images_for_filenames(imgs, labels)
num_categories = len(data_train['Id'].unique())
     
print(f'Number of categories: {num_categories}')
size_buckets = Counter(data_train['Id'].value_counts().values)
plt.figure(figsize=(10, 6))

plt.bar(range(len(size_buckets)), list(size_buckets.values())[::-1])
plt.xticks(range(len(size_buckets)), list(size_buckets.keys())[::-1])
plt.title("Num of categories by images in the training set")

plt.show()

print(data_train['Id'].value_counts().tail(10).keys())
#ploting images with less no. of example
less_image_eg = data_train['Id'].value_counts().tail(10).keys()
print(less_image_eg)
file_name_less = []
label = []
for i in less_image_eg:
    file_name_less.extend(data_train[data_train['Id'] == i]['Image'])
    label.append(i)
print(np.asarray(label).shape)
plot_images_for_filenames(file_name_less , label , rows = 3)
def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True
#is_grey = [is_grey_scale(f'{INPUT_DIR}/train/{i}') for i in data_train['Image'].sample(frac=0.2)]
#grey_perc = round(sum([i for i in is_grey]) / len([i for i in is_grey]) * 100, 2)
#print(f"% of grey images: {grey_perc}")
img_sizes = Counter([Image.open(f'{INPUT_DIR}/train/{i}').size for i in data_train['Image']])

size, freq = zip(*Counter({i: v for i, v in img_sizes.items() if v > 1}).most_common(20))

plt.figure(figsize=(10, 6))

plt.bar(range(len(freq)), list(freq), align='center')
plt.xticks(range(len(size)), list(size), rotation=70)
plt.title("Image size frequencies (where freq > 1)")

plt.show()
from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, transform_matrix_offset_center, img_to_array)
img = Image.open(f'{INPUT_DIR}/train/ff38054f.jpg')
img_arr = img_to_array(img)
print(img_arr.shape)
plt.imshow(img)
#image rotation
imgs = [
    random_rotation(img_arr, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')*255
    for _ in range(5)]
plot_images(imgs, None, rows=1)
imgs = [
    random_shift(img_arr, wrg=0.1, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)
imgs = [
    random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)
imgs = [
    random_zoom(img_arr, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)
import random

def random_greyscale(img, p):
    if random.random() < p:
        return np.dot(img[...,:1], [0.299]).T
    
    return img

imgs = [
    random_greyscale(img_arr, 0.5) * 255
    for _ in range(5)]

plot_images(imgs, None, rows=1)
def augmentation_pipeline(img_arr):
    img_arr = random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_zoom(img_arr, zoom_range=(0.9, 2.0), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    #img_arr = random_greyscale(img_arr, 0.4)

    return img_arr
imgs = [augmentation_pipeline(img_arr) * 255 for _ in range(5)]
plot_images(imgs, None, rows=1)
print(file_name_less)
print(label)
print(data_train.shape)
import scipy.misc
for i in file_name_less:
    img = Image.open(f'{INPUT_DIR}/train/{i}')
    img_arr = img_to_array(img)
   # print(img_arr)
    for j in range(5):
        imgs = augmentation_pipeline(img_arr)*255
        plot_images(imgs , None , rows = 4)
        img_name = str(j) + "_" + str(i)
        print(img_name)
        scipy.misc.imsave('..input/train/{img_name}', image_array)
        data_train['Id'].append(data_train['Id'][data_train['Image'] == i])
        data_train['Image'].append(img_name)
        train.append(imgs)
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from PIL import Image
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
train_images = glob("../input/train/*jpg")
test_images = glob("../input/test/*jpg")
df = pd.read_csv("../input/train.csv")

df["Image"] = df["Image"].map( lambda x : "../input/train/"+x)
ImageToLabelDict = dict( zip( df["Image"], df["Id"]))
SIZE = 64
#image are imported with a resizing and a black and white conversion
def ImportImage( filename):
    img = Image.open(filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]
aug_img = []
for img in file_name_less:
    i = str(INPUT_DIR) + "/train/" + str(img)
    aug_img.append(ImportImage(i))
x_to_be_aug = np.asarray(aug_img)
print(x_to_be_aug)
def plotImages( images_arr, n_images=4):
    fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        if img.ndim != 2:
            img = img.reshape( (SIZE,SIZE))
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()
x_to_be_aug = x_to_be_aug.reshape( (-1,SIZE,SIZE,1))
print(x_to_be_aug.shape)
image_gen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True)


image_gen.fit(x_to_be_aug, augment=True)
print(x_to_be_aug)
new_aug_img = []
for i in x_to_be_aug:
    for _ in range(5):
        img = augmentation_pipeline(i)
        new_aug_img.append(img)
new_aug_img = np.asarray(new_aug_img)
print(new_aug_img.shape)
print(label)
label_new = []
for i in label:
    for _ in range(5):
        label_new.append(i)
        
label_new = np.asarray(label_new)
print(label_new.shape)
train_img = np.array([ImportImage( img) for img in train_images])
x = train_img
x = x.reshape( (-1,SIZE,SIZE,1))
input_shape = x[0].shape
x_train = x.astype("float32")
print(input_shape)
data = pd.read_csv("../input/train.csv")
print(data["Id"])
y_train = data["Id"]
print(y_train)
y_train = np.asarray(y_train)
y_train = np.concatenate((y_train , label_new) , axis = 0)
y_train = np.reshape(y_train , (9900 , 1))
print(y_train.shape)
y_train = pd.DataFrame(y_train)
y_train = pd.get_dummies(y_train)
y_train = np.asarray(y_train)
print(y_train.shape)
print(x_train.shape)
x_train = np.concatenate((x_train , new_aug_img) , axis = 0)
print(x_train.shape)
model = Sequential()
model.add(Conv2D(48, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.33))
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(36, activation='relu'))
model.add(Dense(4251, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(x_train , y_train , batch_size = 128 , epochs = 10 , validation_split=0.2)
print(model.summary())
