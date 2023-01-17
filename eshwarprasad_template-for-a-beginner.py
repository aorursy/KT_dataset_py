# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

%matplotlib inline
path = Path("kaggle/input/siim-isic-melanoma-classification/")

train_path = path / 'train'

test_path = path / 'test'



print(train_path)

print(test_path)
!ls /kaggle/input/siim-isic-melanoma-classification/
img_train_path = path / 'jpeg' / 'train'

img_test_path = path / 'jpeg' / 'test'



print(img_train_path)

print(img_test_path)
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



from keras import models, regularizers, layers, optimizers, losses, metrics

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Conv3D

from keras.utils import np_utils, to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
def image_show(img_num, img_folder, img_size):

    

    img_ind = 'ISIC'

    img_name = '{}_{}'.format(img_ind, img_num)

    

    if img_folder == 'train':

        img_dir = img_train_path

    elif img_folder == 'test':

        img_dir = img_test_path

        

    img_path = str(img_dir)+'/'+str(img_name)+'.jpg'

    

    print("Image Path", img_path)

    

    img = image.load_img(img_path, target_size = (img_size, img_size))

    imgplot = plt.imshow(img)

    print(img_ind, "Image Number", img_num)

    plt.show()
image_show('0074542', 'train', 224)
#importing CSV Dataset

train_path = path / 'train.csv'

test_path = path / 'test.csv'

train = pd.read_csv(train_path)

test  = pd.read_csv(test_path)



train.shape, test.shape
train.isna().sum()
train['sex'] = train['sex'].fillna('na')

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

#Replacing Null age values with the mean age of the training_set



train['age_approx'] = train['age_approx'].fillna(int(train['age_approx'].mean()))
test.isna().sum()
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
train.head()