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



# Load other libraries

import bcolz

import random

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.axes_grid1 import ImageGrid
df_train = pd.read_csv('../input/labels.csv')

df_train.head(10)
im_size = 300

x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)

y_train = []
for i,j in tqdm(df_train.values):

    image = load_img('../input/train/{}.jpg'.format(i), target_size=(im_size, im_size))

    x_train_raw.append(img_to_array(image))

    y_train.append(j)
print("Number of images: ",len(y_train),"\n",

      "Array shape: ",x_train_raw.shape)

plt.imshow(x_train_raw[1]/255)
def img_augmentation(x_train_raw, y_train):

    n = x_train_raw.shape[0]

    rand = random.sample(range(0,n),2)

    img_container = np.zeros((2,int(x_train_raw.shape[1]*0.8),int(x_train_raw.shape[2]*0.8),3),dtype=np.float32)

    data_generator = ImageDataGenerator()

#     plt.ion()

    for i in range(0,len(rand)):

    # Crop Image

        if random.randint(1,101)<20:

            # Top Left

            img_container[i]=x_train_raw[rand[i],0:240,0:240,:]

        elif random.randint(1,101)<40:

            # Top Right

            img_container[i]=x_train_raw[rand[i],0:240,60:300,:]

        elif random.randint(1,101)<60:

            # Bottom Left

            img_container[i]=x_train_raw[rand[i],60:300,0:240,:]

        elif random.randint(1,101)<60:

            # Bottom Right

            img_container[i]=x_train_raw[rand[i],60:300,60:300,:]

        else:

            # Center

            img_container[i]=x_train_raw[rand[i],30:270,30:270,:]

        # Flip Image

        if random.randint(1,101) < 50: 

            flip_horizontal = True

        else:

            flip_horizontal = False

        if random.randint(1,101) < 50: 

            flip_vertical = True

        else:

            flip_vertical = False

        img_container[i] = data_generator.apply_transform(img_container[i],{

            'flip_horizontal':flip_horizontal,

            'flip_vertical':flip_vertical

        })

        print("Original Image:")

        plt.title(y_train[rand[i]])

        plt.imshow(x_train_raw[rand[i],]/255.0)

        plt.show()    

    def plotImages(images_arr, n_images=2):

        fig, axes = plt.subplots(n_images-1, n_images, figsize=(12,12))

        axes = axes.flatten()

        for img, ax in zip( images_arr, axes):

            ax.imshow(img)

            ax.set_xticks(())

            ax.set_yticks(())

            plt.tight_layout()

    plotImages(img_container[:,]/255.)

    
img_augmentation(x_train_raw, y_train)