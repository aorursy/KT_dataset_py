# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import tensorflow_addons as tfa

import cv2



from colorama import Fore, Back, Style

from tensorflow import keras

from tensorflow.keras import layers



y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)



AUTOTUNE = tf.data.experimental.AUTOTUNE

    

print(tf.__version__)
monet_jpg_directory = '../input/gan-getting-started/monet_jpg/'

photo_jpg_directory = '../input/gan-getting-started/photo_jpg/'
def getImagePaths(path):

    image_names = []

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            fullpath = os.path.join(dirname, filename)

            image_names.append(fullpath)

    return image_names
monet_images_path = getImagePaths(monet_jpg_directory)

photo_images_path = getImagePaths(photo_jpg_directory)
print(f"{y_}Number of Monet images: {g_} {len(monet_images_path)}\n")

print(f"{y_}Number of Photo images: {g_} {len(photo_images_path)}\n")
def getShape(images_paths):

    shape = cv2.imread(images_paths[0]).shape

    for image_path in images_paths:

        image_shape=cv2.imread(image_path).shape

        if (image_shape!=shape):

            return "Different image shape"

        else:

            return "Same image shape " + str(shape)
getShape(monet_images_path)
getShape(photo_images_path)
def display_multiple_img(images_paths, rows, cols):

    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,8) )

    for ind,image_path in enumerate(images_paths):

        image=cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        try:

            ax.ravel()[ind].imshow(image)

            ax.ravel()[ind].set_axis_off()

        except:

            continue;

    plt.tight_layout()

    plt.show()
display_multiple_img(monet_images_path, 4, 4)
display_multiple_img(photo_images_path, 4, 4)