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



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

from mpl_toolkits.axes_grid1 import ImageGrid

import re
im_size     = 300

train_files = glob('../input/dog-breed-identification/train/*.jpg')

INPUT_LOC   = '../input/dog-breed-identification/train/'



## Initialize Bcolz array 

bc_img_arr = bcolz.zeros((0,im_size,im_size,3),np.float32)



## Build Bcolz Array  - TRAIN 

for img in tqdm(train_files):

    file_name_jpg = re.search("(\w+.jpg)", img)

    if file_name_jpg:

        file_name = file_name_jpg.group(0)

        image = load_img(INPUT_LOC + file_name, target_size=(im_size, im_size))

        image = img_to_array(image)

        bc_img_arr.append(image)

    else:

        print("file Not found")



print("Number of Training Images".ljust(40) + ":" + str(len(train_files)))

print("Length of Bcolz Array".ljust(40) + ":" + str(len(bc_img_arr)))


######### CROP FUNCTION #########

def crop_by_position(ORIGINAL_IMG, RESIZE_FACTOR, POSITION = "Top_Left"):

    

    height, width, z      = ORIGINAL_IMG.shape

    new_height, new_width = int(height * RESIZE_FACTOR), int(width * RESIZE_FACTOR)

    if new_height > height or new_width > width: print("Crop cannot be performed, New size bigger than original")

    

    if POSITION == "Original":

        return ORIGINAL_IMG/255

    if POSITION == "Top_Left":

        return ORIGINAL_IMG[:new_height            , :new_width         , : ]/255.

    elif POSITION == "Top_Right":

        return ORIGINAL_IMG[:new_height            , (width-new_width): , : ]/255.

    elif POSITION == "Bottom_Left":

        return ORIGINAL_IMG[(height - new_height): , :new_width         , : ]/255.

    elif POSITION == "Bottom_Right":

        return ORIGINAL_IMG[(height - new_height): , (width-new_width): , : ]/255.

    elif POSITION == "Center":

        startx, starty = width//2-(new_width//2), height//2-(new_height//2)

        return ORIGINAL_IMG[starty:starty+new_height,startx:startx+new_width]/255

    else:

        return False



######### IMAGE AUGMENTATION #########    

def Apply_Image_Augementation(BATCH, RESIZE_FACTOR=0.8):

    pos_lst       = ["Original", "Top_Left", "Top_Right", "Bottom_Left", "Bottom_Right", "Center", "Flip_Vertical", "Flip_Horizontal"]

    RNO           = random.randint(1,101)

    RESIZE_FACTOR = 0.7

    

    ## Operate on a batch

    for pic in BATCH:



        ## Display all transformations ##

        f, ax = plt.subplots(1, 8, figsize=(15,15)) 

        for ax_, pos in zip(ax, pos_lst): 

            if   pos == "Flip_Vertical"   : img = ImageDataGenerator().apply_transform(pic,{'flip_vertical':True})/255

            elif pos == "Flip_Horizontal" : img = ImageDataGenerator().apply_transform(pic,{'flip_horizontal':True})/255

            else                          : img = crop_by_position(pic, RESIZE_FACTOR, pos)



            ax_.imshow(img)

            ax_.set_title(pos)

            ax_.axis("off")

            

        ## Display transformations by given probability ##

        f, ax = plt.subplots(1, 2, figsize=(4,4)) 

        img_selected = crop_by_position(pic, RESIZE_FACTOR, "Top_Left")                           if RNO < 20                else pic

        img_selected = crop_by_position(img_selected, RESIZE_FACTOR, "Top_Right")                 if RNO >= 20 and RNO < 40  else img_selected

        img_selected = crop_by_position(img_selected, RESIZE_FACTOR, "Bottom_Left")               if RNO >= 40 and RNO < 60  else img_selected

        img_selected = crop_by_position(img_selected, RESIZE_FACTOR, "Bottom_Right")              if RNO >= 60 and RNO < 80  else img_selected

        img_selected = crop_by_position(img_selected, RESIZE_FACTOR, "Center")                    if RNO >= 80               else img_selected

        img_selected = ImageDataGenerator().apply_transform(img_selected,{'flip_vertical':True})  if RNO < 50                else img_selected

        img_selected = ImageDataGenerator().apply_transform(img_selected,{'flip_vertical':True})  if RNO >= 50               else img_selected



        ax[0].imshow(pic/255.)

        ax[0].axis("off")

        ax[0].set_title("-- ORIGINAL --")

        ax[1].imshow(img_selected)

        ax[1].axis("off")

        ax[1].set_title("-- AUGMENTED  --")

        



def Get_Batches(IMAGE_ARR, BATCH_SIZE):

    ## Initialize ImageDataGenerator & divide them for batches

    datagen = ImageDataGenerator()

    

    ## Create Batches

    batches = datagen.flow(bc_img_arr, shuffle=True, batch_size=2)

    print("Total Batches : " + str(len(batches)), "\n")

    

    ## Generator to work on batches

    for batch in batches:

        yield batch    

        
## Create BAtches from BCOLZ Array

batches = Get_Batches(IMAGE_ARR=bc_img_arr, BATCH_SIZE=2)
## How many batches to examine - This is used to control the number of iterations

batch_id = 1



## Iterate through batches, with each batch containing "BATCH_SIZE" number of images

for idx, batch in enumerate(batches):

    

    # Print

    print("Batch Number : ", idx + 1)

    

    ## Augment Images in batch number

    Apply_Image_Augementation(batch)

    

    ## Exit condition

    if idx+1 == batch_id: break

        

        