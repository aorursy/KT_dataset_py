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
#modules needed



import os

import gc

import cv2

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import *

from tqdm import tqdm



#Help with pre-processing data from:

'''

https://www.kaggle.com/koshirosato/bee-or-wasp-base-line-using-resnet50

https://www.kaggle.com/mahmoudvaziri/wasp-or-bee-ensemble

'''
imagesPath = '../input/bee-vs-wasp/kaggle_bee_vs_wasp/'

df = pd.read_csv('../input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv')

df = df[df.photo_quality==1]

df.head()
#Replacing '\' with '/'



for idx in tqdm(df.index):    

    df.loc[idx,'path']=df.loc[idx,'path'].replace('\\', '/') 

    

df.head()
#Visualisation of data

labels = list(df['label'].unique())

y = list(df['label'].value_counts())

plt.pie(y, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title('Unique values of the original data')

plt.show()
#Count of images that are high quality

df = df.query('photo_quality == 1')

df['label'].value_counts()
#checking for training, validation and test data in data frame, 

train_df = df.query('is_validation == 0 & is_final_validation == 0').reset_index(drop=True)

val_df = df.query('is_validation == 1').reset_index(drop=True)

test_df = df.query('is_final_validation == 1').reset_index(drop=True)
train_df.head()
val_df.head()
test_df.head()
'''

Copied from:

https://www.kaggle.com/koshirosato/bee-or-wasp-base-line-using-resnet50

'''

IMG_SIZE = 256



#Creating Datasets

def create_datasets(df, img_size):

    imgs = []

    for path in tqdm(df['path']):

        img = cv2.imread(imagesPath+path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (img_size,img_size))

        imgs.append(img)

        

    imgs = np.array(imgs, dtype='float32')

    imgs = imgs / 255.0

    df = pd.get_dummies(df['label'])

    return imgs, df





train_imgs, train_df = create_datasets(train_df, IMG_SIZE)

val_imgs, val_df = create_datasets(val_df, IMG_SIZE)

test_imgs, test_df = create_datasets(test_df, IMG_SIZE)
train_df.head()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D
train_imgs.shape[1:]
#Building the CNN Model



model = Sequential()



model.add(Conv2D(256, (3,3), input_shape=train_imgs.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(64))

model.add(Activation('relu'))



model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])



model.summary()
#Training the CNN

model.fit(train_imgs, train_df, batch_size = 32, epochs = 5, validation_data =(val_imgs, val_df))
model.evaluate(test_imgs, test_df)