# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import keras.backend as B

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D , MaxPool2D , Dense , Flatten , Activation

from keras.models import Sequential , Model

from keras.optimizers import Adagrad , Adam , SGD

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import os

from pathlib import Path

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
TRAIN_path = "../input/seg_train/seg_train/"

TEST_path = "../input/seg_test/seg_test/"
category = [ 'buildings' , 'forest' , 'glacier' , 'mountain' , 'sea', 'street']



for cat in category:

    print("The total pics in {} folder are {}".format(cat , len(os.listdir(os.path.join(TRAIN_path , cat)))))
def convframe(path , category):

    

    file_list= []

    for cat in category:

        for file in os.listdir(os.path.join(path , cat)):

            file_list.append([os.path.join(path,cat) +'/'+file, cat])

    df = pd.DataFrame(file_list , columns=['Images_file' , 'category_label'])

    

    return df

    
# train dataframe

train_df = convframe(TRAIN_path , category)



# test dataframe

test_df = convframe(TEST_path , category)
train_df.shape , test_df.shape
def class_plot(df , label):



    fig , axes = plt.subplots(nrows=3 , ncols=3 , figsize=(10,10))

    fetch = df.loc[df.category_label==label][:10]

    fetch.reset_index(inplace=True)

    n =0

    for i in range(0,3):

        for j in range(0,3):

            img=plt.imread(fetch['Images_file'][n])

            axes[i,j].imshow(img)

            n+=1    

class_plot(train_df , 'buildings')
class_plot(train_df , 'mountain')
class_plot(train_df , 'sea')
class_plot(train_df , 'glacier')
class_plot(train_df , 'street')
class_plot(train_df , 'forest')