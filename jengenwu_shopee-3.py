# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow_hub as hub



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import cv2 # read image

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
PATH = "/kaggle/input/shopee-product-detection-student/"

train_dir = os.path.join(PATH, "train/train/train")

test_dir = os.path.join(PATH, "test/test/test")
train_dir
train_list = os.listdir(train_dir)

test_list = os.listdir(test_dir)
df_train = pd.read_csv(os.path.join(PATH, "train.csv"))

df_train
df_test = pd.read_csv(os.path.join(PATH, "test.csv"))

df_test
dir_cat = os.path.join(train_dir,"00")

first_image_name = os.listdir(dir_cat)[0]

first_im = cv2.imread(os.path.join(dir_cat, first_image_name))

print(type(first_im))

print(first_im.shape) # image height and width

print(first_im.min(),first_im.max()) # range from 0 to 255 

plt.imshow(first_im)

plt.show
sns.countplot(df_train["category"])
batch_size = 128

epoch = 15

img_h = 150

img_w = 150
datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)

train_iter = datagen.flow_from_directory(batch_size = batch_size,

                                        directory=train_dir,

                                        shuffle=True,

                                        target_size=(img_h, img_w),

                                        class_mode="categorical",

                                        subset = "training")



val_iter = datagen.flow_from_directory(batch_size = batch_size,

                                        directory=train_dir,

                                        shuffle=True,

                                        target_size=(img_h, img_w),

                                        class_mode="categorical",

                                        subset = "validation")
test_iter = datagen.flow_from_directory(

    directory = test_dir,

    shuffle=False,

    target_size=(img_h, img_w)

)
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

feature_extractor = hub.KerasLayer(URL,input_shape=(img_h,img_w,3))
base_model.summary()