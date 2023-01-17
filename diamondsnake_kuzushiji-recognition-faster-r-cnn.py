# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

#from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten#, Dropout, Activation

from keras.utils import to_categorical
import tensorflow

import keras

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import patches

#import opencv-python

import sklearn

import h5py
# import data

train_df = pd.read_csv("/kaggle/input/kuzushiji-recognition/train.csv")

train_df.head()
# reformat with just the first label 

new_df = train_df.drop('labels', 1).join(train_df["labels"].str.split(" ", n=5, expand = True).drop(5,1)).fillna(0)

new_df.columns = ['image_id','label','xmin','ymin','width','height']

new_df['xmin'] = new_df['xmin'].astype(int)

new_df['ymin'] = new_df['ymin'].astype(int)

new_df['width'] = new_df['width'].astype(int)

new_df['height'] = new_df['height'].astype(int)

new_df.head()
# show training image and boxes



fig = plt.figure()



#add axes to the image

ax = fig.add_axes([0,0,1,1])



# read and plot the image

top_image = plt.imread('/kaggle/input/kuzushiji-recognition/train_images/100241706_00004_2.jpg')

plt.imshow(top_image)



# iterating over the image for different objects

for _,row in new_df[new_df.image_id == "100241706_00004_2"].iterrows():

    xmin = row.xmin

    ymin = row.ymin

    width = row.width

    height = row.height

    

    # add bounding boxes to the image

    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = 'r', facecolor = 'none')

    ax.add_patch(rect)
data = pd.DataFrame()

data['format'] = train_df['image_id']



# as the images are in train_images folder, add train_images before the image name

for i in range(data.shape[0]):

    data['format'][i] = 'train_images/' + data['format'][i]



# add xmin, ymin, xmax, ymax and class as per the format required

for i in range(data.shape[0]):

    data['format'][i] = data['format'][i] + ',' + str(new_df['xmin'][i]) + ',' + str(new_df['ymin'][i]) + ',' + str(new_df['width'][i]) + ',' + str(new_df['height'][i]) + ',' + str(new_df['label'][i])



data.to_csv('annotate.txt', header=None, index=None, sep=' ')
data.head()