%matplotlib inline



import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from PIL import Image



train_dir = '../input/train/'

test_dir = '../input/test/'



len(os.listdir(test_dir))
# Extract images

img_size = 64

channels_num = 1



def extract_images(dr):

    num_images = len(os.listdir(dr))

    data = np.ndarray([num_images, img_size, img_size, channels_num], np.float32)

    

    for i, img in enumerate(os.listdir(dr)):

        img = Image.open(dr + img)

        img_gray = img.convert('L')

        img_resized = img_gray.resize((img_size, img_size), Image.ANTIALIAS)

        img_np = np.array(img_resized)

        img_normal = (img_np - (255.0 / 2.0)) / 255.0

        data[i,:,:,:] = img_normal.reshape((img_size, img_size, channels_num))

        

    return data

    

#train_images = extract_images(train_dir)

test_images = extract_images(test_dir)
%matplotlib inline

plt.show(plt.imshow(test_images[0].reshape(img_size, img_size), cmap=plt.cm.Greys))