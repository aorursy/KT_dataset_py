# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm



import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir('/kaggle/input/gan-getting-started')     





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
path, dirs, files = next(os.walk("../input/gan-getting-started/monet_jpg"))

print(f"Monet:{len(files)}")

path, dirs, files = next(os.walk("../input/gan-getting-started/monet_tfrec"))

print(f"Monet_tfrec:{len(files)}")

path, dirs, files = next(os.walk("../input/gan-getting-started/photo_tfrec"))

print(f"Photo_tfrec:{len(files)}")

path, dirs, files = next(os.walk("../input/gan-getting-started/photo_jpg"))

print(f"Photo:{len(files)}")
def getImagePaths(path):

    image_names = []

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            fullpath = os.path.join(dirname, filename)

            image_names.append(fullpath)

    return image_names
photo_files=getImagePaths("../input/gan-getting-started/photo_jpg")

monet_files=getImagePaths("../input/gan-getting-started/monet_jpg")
from PIL import Image   

# creating a object  

im = Image.open(r"../input/gan-getting-started/monet_jpg/00068bc07f.jpg")  

im.show() 
print(cv2.imread(photo_files[0]).shape)

print(cv2.imread(monet_files[0]).shape)
def displayImages(images_paths, rows, cols):

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
displayImages(photo_files,2,4)
displayImages(monet_files,2,4)
tf.__version__
IMAGE_SIZE = (256, 256)

monet = []

photos=[]

for files in tqdm(monet_files):

    image = cv2.imread(files)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE) 

    monet.append(image)
for files in tqdm(photo_files):

    image = cv2.imread(files)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE) 

    photos.append(image)
photos = np.array(photos, dtype = 'float32')

monet = np.array(monet, dtype = 'float32')