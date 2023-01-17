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
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import os

import numpy as np

import matplotlib.pyplot as plt
tf.__version__
import zipfile

with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:

   # Extract all the contents of zip file in current directory

   zip_obj.extractall('/kaggle/working/')

    

print('After zip extraction:')

print(os.listdir("/kaggle/working/"))
! ls plates/train/
! ls plates/train/cleaned | head 
! ls plates/train/dirty | head 
PATH =  'plates'

train_dir = os.path.join(PATH, 'train')

train_dir
train_dirty_dir = os.path.join(train_dir, 'dirty')  



train_cleaned_dir = os.path.join(train_dir, 'cleaned')  

train_dirty_dir, train_cleaned_dir
num_dirty_tr = len(os.listdir(train_dirty_dir))

num_cleaned_tr = len(os.listdir(train_cleaned_dir))

num_dirty_tr, num_cleaned_tr
total_train = num_dirty_tr + num_cleaned_tr

total_train
batch_size = 10

epochs = 200

IMG_SIZE = 160
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=15,

                    width_shift_range=.1,

                    height_shift_range=.1,

                    horizontal_flip=True,

                    zoom_range=0.1, 

                    brightness_range=[0.8,1.0]

                    )
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,

                                                     directory=train_dir,

                                                     shuffle=True,

                                                     target_size=(IMG_SIZE, IMG_SIZE),

                                                     class_mode='binary', 

)