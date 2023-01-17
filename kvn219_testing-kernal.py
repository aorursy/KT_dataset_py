# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or 

# pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/train_2"]).decode("utf8"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image, ImageFilter

import random

import cv2

import os, glob



#t = pd.read_csv('../input/train_info.csv'); t.head()

#s = pd.read_csv('../input/submission_info.csv'); s.head()

train_files = [f for f in glob.glob("../input/train_2/*")]

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in train_files[:100]:

    im = cv2.imread(l)

    im = cv2.resize(im, (50, 50)) 

    plt.subplot(10, 10, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
import tensorflow as tf

from scipy import ndimage
def load_images(folder, min_num_images):

    """Load the data for a single letter label."""

    image_files = os.listdir(folder)

    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)

    print(folder)

    num_images = 0

    for image in image_files:

        image_file = os.path.join(folder, image)

    try:

        image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth

        dataset[num_images, :, :] = image_data

        num_images = num_images + 1

    except IOError as e:

        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    

    dataset = dataset[0:num_images, :, :]

    print('Full dataset tensor:', dataset.shape)

    print('Mean:', np.mean(dataset))

    print('Standard deviation:', np.std(dataset))

    return dataset
datset = load_images("../input/train_2", 1)
import glob

image_list = []

for filename in glob.glob('../input/train_2/*.jpg'): #assuming gif

    im=Image.open(filename)

    image_list.append(im)


len(image_list)
t = pd.read_csv('../input/train_info.csv'); t.head()