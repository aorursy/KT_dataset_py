# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.applications.vgg19 import VGG19

from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten,  Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import load_img

#from keras.applications.vgg16 import preprocess_input

from keras.applications.resnet50 import preprocess_input

from keras.preprocessing.image import img_to_array

import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import cv2

import sys

import bcolz

import random





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/labels.csv')

df_test = pd.read_csv('../input/sample_submission.csv')
df_train.head(10)
%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

from mpl_toolkits.axes_grid1 import ImageGrid
train_files = glob('../input/train/*.jpg')

test_files = glob('../input/test/*.jpg')

train_files[1]
plt.imshow(plt.imread(train_files[100]))
targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)

one_hot_labels[0]
im_size = 300
y_train = []

y_val = []

x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)

x_val_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
y_train_label=[]

i = 0 

for f, breed in tqdm(df_train.values):

    # load an image from file

    image = load_img('../input/train/{}.jpg'.format(f), target_size=(im_size, im_size))

    image = img_to_array(image)

    # prepare the image for the VGG model

    #image = preprocess_input(image)

    label = one_hot_labels[i]

    if random.randint(1,101) < 80: 

        x_train_raw.append(image)

        y_train.append(label)

        y_train_label.append(breed)

    else:

        x_val_raw.append(image)

        y_val.append(label)

    i += 1
y_train_raw = np.array(y_train, np.uint8)

y_val_raw = np.array(y_val, np.uint8)

#del(y_train,y_val)

import gc

gc.collect()
print(x_train_raw.shape)

print(y_train_raw.shape)

print(x_val_raw.shape)

print(y_val_raw.shape)
def plotImages( images_arr, n_images=4):

    fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.set_xticks(())

        ax.set_yticks(())

    plt.tight_layout()

plotImages(x_train_raw[0:16,]/255.)
num_class = y_train_raw.shape[1]

num_class
batch_size=2

resize_factor=0.8
def augmentation(x_train_raw,y_train_raw,batch_size,resize_factor):

    

    datagen = ImageDataGenerator()

    data_aug = np.zeros((batch_size,int(im_size*resize_factor),int(im_size*resize_factor),3),dtype=np.float32)



    rand_samp=random.sample(list(np.arange(0,x_train_raw.shape[0])),batch_size)

    print (rand_samp)



#cropping

    for i in range(0,len(rand_samp)):

    #top_left

        if random.randint(1,101) < 20:

            data_aug[i]=x_train_raw[rand_samp[i],0:int(im_size*resize_factor),

                                    0:int(im_size*resize_factor), : ]

            print ('top_left')

            

    #top_right    

        elif random.randint(1,101) < 20:

            data_aug[i]=x_train_raw[rand_samp[i],0:int(im_size*resize_factor),

                                im_size-int(im_size*resize_factor):im_size, : ]

            print ('top_right')

            

    #bottom_left 

        elif random.randint(1,101) < 20:

            data_aug[i]=x_train_raw[rand_samp[i],im_size-int(im_size*resize_factor):im_size,

                                0:int(im_size*resize_factor), : ]

            print ('bottom_left')

            

    #bottom_right

        elif random.randint(1,101) < 20:

            data_aug[i]=x_train_raw[rand_samp[i],im_size-int(im_size*resize_factor):im_size,

                                im_size-int(im_size*resize_factor):im_size, : ]

            print ('bottom_right')

                

    #center

        else:

            data_aug[i]=x_train_raw[rand_samp[i],30:im_size-30,30:im_size-30, : ]

            print ('center')

        

#flipping

        if random.randint(1,101) < 50: 

            flip_horizontal = True

        else:

            flip_horizontal = False

        if random.randint(1,101) < 50: 

            flip_vertical = True

        else:

            flip_vertical = False

  

        data_aug[i] = datagen.apply_transform(data_aug[i],{

            'flip_horizontal':flip_horizontal,

            'flip_vertical':flip_vertical,

            })



#displaying the actual images

#     plt.title(y_train_label[rand_samp[i]])

#     plt.imshow(x_train_raw[rand_samp[i],]/255.0)



#displaying the cropped and flipped images    

    def plotImages(images_arr, n_images=2):

        fig, axes = plt.subplots(n_images-1, n_images, figsize=(12,12))

        axes = axes.flatten()

        for img, ax in zip( images_arr, axes ):

            ax.imshow(img)

            ax.set_xticks(())

            ax.set_yticks(())

            plt.tight_layout()

            

    plotImages(x_train_raw[rand_samp,]/255.)    #actaul images

    plotImages(data_aug[:,]/255.)               #cropped images

        

augmentation(x_train_raw, y_train_raw,batch_size,resize_factor)