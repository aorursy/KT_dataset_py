import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
medium_colored_data = pd.read_csv('../input/hmnist_28_28_RGB.csv')
medium_colored_data.head()
medium_colored_data.shape
example = medium_colored_data.drop("label", axis=1).values[0]
to_show = example.reshape((28,28,3))

fig, ax = plt.subplots(1,4,figsize=(20,5))
for channel in range(3):
    ax[channel].imshow(to_show[:,:,channel], cmap="gray")
    ax[channel].set_title("Channel {}".format(channel+1))
    ax[channel].set_xlabel("Width")
    ax[channel].set_ylabel("Height")
ax[3].imshow(to_show)
ax[3].set_title("All channels together")
ax[3].set_xlabel("Width")
ax[3].set_ylabel("Height")
order_example = np.arange(0,12)
order_example
show_order = order_example.reshape(2,2,3)
print(show_order[:,:,0])
print(show_order[:,:,1])
print(show_order[:,:,2])
medium_data = pd.read_csv('../input/hmnist_28_28_L.csv')
medium_data.head()
print(medium_data.shape)
def show_inarow(data, row_shape):
    example = data.drop("label", axis=1).values[0:4]
    to_show = example.reshape(row_shape)
    fig, ax = plt.subplots(1,4,figsize=(20,5))
    for image_example in range(4):
        ax[image_example].imshow(to_show[image_example,:,:], cmap="gray")
        ax[image_example].set_title("Grayscaled image {}".format(image_example))
        ax[image_example].set_xlabel("Widht")
        ax[image_example].set_ylabel("Height")
show_inarow(medium_data, (4,28,28))
big_data = pd.read_csv("../input/hmnist_64_64_L.csv")
big_data.head()
big_data.shape
show_inarow(big_data, (4,64,64))
small_data = pd.read_csv("../input/hmnist_8_8_L.csv")
small_colored_data = pd.read_csv("../input/hmnist_8_8_RGB.csv")
small_data.head()
print(small_data.shape)
print(small_colored_data.shape)
show_inarow(small_data, (4,8,8))
show_inarow(small_colored_data, (4,8,8,3))
from os import listdir

classes_dir = listdir("../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000")
classes_dir
files = listdir("../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/01_TUMOR")
for n in range(5):
    print(files[n])
from scipy.misc import imread

def show_set(basepath, classes_dir, num_file):
    fig, ax = plt.subplots(2,4,figsize=(20,10))
    for n in range(4):
        for m in range(2):
            class_idx = m * 4 + n
            path = basepath + classes_dir[class_idx] + "/"
            files = listdir(path)
            image = imread(path + files[num_file])
            ax[m,n].imshow(image)
            ax[m,n].set_title(classes_dir[class_idx])
basepath = "../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/"
show_set(basepath, classes_dir, num_file=0)
basepath = "../input/kather_texture_2016_larger_images_10/Kather_texture_2016_larger_images_10/"
files = listdir(basepath)
files
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(imread(basepath + files[0]))
ax[1].imshow(imread(basepath + files[1]))