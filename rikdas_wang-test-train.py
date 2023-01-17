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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input

from keras.models import Model

from keras import backend as K
from glob import glob

imagePatches = glob('../input/wang_test_train/wang_test_train/train/*.jpg', recursive=True)

imagePatches_1 = glob('../input/wang_test_train/wang_test_train/test/*.jpg', recursive=True)

train_images = imagePatches

test_images = imagePatches_1

for filename in imagePatches[0:]:

    print (filename)
for filename in imagePatches_1[0:]:

    print (filename)
from keras.preprocessing import image

import pandas as pd

#from keras import backend as K

imagenumber = []

images = []

for img in train_images:

        imagenumber.append(img)

        img_data = image.load_img(img, target_size=(128, 128))

        img_data = image.img_to_array(img_data)

        #img_pixel_val = img_data

        images.append(img_data)

training_images = np.array(images)

nmbr = np.array(imagenumber)

img_no = pd.DataFrame(nmbr)

training_images.shape
training_images
from keras.preprocessing import image

import pandas as pd

#from keras import backend as K

imagenumber_1 = []

images_1 = []

for img in test_images:

        imagenumber_1.append(img)

        img_data = image.load_img(img, target_size=(128, 128))

        img_data = image.img_to_array(img_data)

        #img_pixel_val = img_data

        images_1.append(img_data)

testing_images = np.array(images_1)

nmbr = np.array(imagenumber_1)

img_no = pd.DataFrame(nmbr)

testing_images.shape
input_img = Input(shape=(128, 128, 3))



# Ecoding

conv1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)

pool1 = MaxPooling2D(pool_size=(2,2), padding='same')(conv1)

conv2 = Conv2D(8,(3, 3), padding='same', activation='relu')(pool1)

pool2 = MaxPooling2D(pool_size=(2,2), padding='same')(conv2)

conv3 = Conv2D(8,(3, 3), padding='same', activation='relu')(pool2)

pool3 = MaxPooling2D(pool_size=(2,2), padding='same')(conv3)



# Decoding

conv4 = Conv2D(8,(3, 3), padding='same', activation='relu')(pool3)

up1 = UpSampling2D((2, 2))(conv4)

conv5 = Conv2D(8,(3, 3), padding='same', activation='relu')(up1)

up2 = UpSampling2D((2, 2))(conv5)

conv6 = Conv2D(3,(3, 3), padding='same', activation='sigmoid')(up2)

up3 = UpSampling2D((2, 2))(conv6)



autoencoder = Model(input_img,up3)

autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

autoencoder.summary()
autoencoder.fit(training_images, training_images,

                epochs=2000,

                batch_size=10,

                shuffle=True,

                validation_data=(testing_images, testing_images)) 
encoder = Model(input_img, pool3)
features = encoder.predict(testing_images)

x = features.reshape(100,2048)

print(x.shape)
x
from sklearn.preprocessing import MinMaxScaler

data = x

scaler = MinMaxScaler()

scaler.fit(data)

features_norm_wang = scaler.transform(data)

features_wang = pd.DataFrame(features_norm_wang)
features_wang
df = pd.DataFrame(features_wang)

img_no_2 = pd.DataFrame(nmbr)



#filepath_1 = 'wang_image_no_2.csv'

filepath_2 = 'wang_autoencoder_norm_train.csv'



#img_no_2.to_csv(filepath_1, index = False)

df.to_csv(filepath_2, index = False)