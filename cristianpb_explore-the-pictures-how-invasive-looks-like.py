import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import matplotlib.pyplot as plt

from keras.preprocessing import image

import os

import PIL
train_labels = pd.read_csv('../input/train_labels.csv')

train_labels.head()
path_prefix = '../input/train/'

train_path = glob(path_prefix + '*')

train_number = [os.path.splitext(f.split("/")[-1])[0] for f in train_path]
train_path[:5]
train_number[:5]
grid_size = 6

f, axarr = plt.subplots(grid_size, grid_size, figsize=(16, 16))

f.subplots_adjust(hspace=0.0,wspace=0.0)

plt.rcParams.update({'font.size': 22})



for k,photo_name in enumerate(train_path):

    #print(k,photo_name)

    if k > grid_size*grid_size-1: break

    img_path = photo_name

    img_number = int(os.path.splitext(photo_name.split("/")[-1])[0])

    img_label = train_labels[train_labels['name'] == img_number].invasive.values[0]

    #print(img_label)

    img = image.load_img(img_path, target_size=(300, 300))

    img = image.img_to_array(img)

    axarr[int(k/grid_size), k%grid_size].imshow(img / 255.)

    axarr[int(k/grid_size), k%grid_size].axis('off')

    if img_label == 1:

        axarr[int(k/grid_size), k%grid_size].text(20, 40, img_label, bbox={'facecolor':'red', 'alpha':0.5})

    elif img_label == 0:

        axarr[int(k/grid_size), k%grid_size].text(20, 40, img_label, bbox={'facecolor':'blue', 'alpha':0.5})



    

plt.show()
sizes = [PIL.Image.open(name_photo).size for name_photo in train_path]
x_width = [f[0] for f in sizes]

y_height = [f[1] for f in sizes]
plt.rcParams.update({'font.size': 10})

plt.hist(x_width, bins=1, label='width')

plt.hist(y_height, bins=1, label='height')

plt.legend(loc='best')

plt.show()