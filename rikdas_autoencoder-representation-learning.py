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
import matplotlib.pyplot as plt



### Autoencoder ###

import tensorflow as tf

import tensorflow.keras



from tensorflow.keras import models, layers

from tensorflow.keras.models import Model, model_from_json



from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input
from glob import glob

imagePatches = glob('../input/wang_test_train/wang_test_train/train/*.jpg', recursive=True)

imagePatches_1 = glob('../input/wang_test_train/wang_test_train/test/*.jpg', recursive=True)

train_images = imagePatches

test_images = imagePatches_1

for filename in imagePatches[0:]:

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

        images.append(img_data.flatten())

training_images = np.array(images)

nmbr = np.array(imagenumber)

img_no = pd.DataFrame(nmbr)

training_images.shape
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

        images_1.append(img_data.flatten())

testing_images = np.array(images_1)

nmbr = np.array(imagenumber_1)

img_no = pd.DataFrame(nmbr)

testing_images.shape
input_size = 49152

hidden_size_1 = 500

hidden_size_2 = 100

code_size = 50



input_img = Input(shape=(input_size,))

hidden_1 = Dense(hidden_size_1, activation='relu')(input_img)

hidden_2 = Dense(hidden_size_2, activation='relu')(hidden_1)

code = Dense(code_size, activation='relu')(hidden_2)

hidden_3 = Dense(hidden_size_2, activation='relu')(code)

hidden_4 = Dense(hidden_size_1, activation='relu')(hidden_3)

output_img = Dense(input_size, activation='sigmoid')(hidden_4)



autoencoder = Model(input_img, output_img)

autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
autoencoder.fit(training_images, training_images,

                epochs=400,

                batch_size=30,

                shuffle=True,

                validation_data=(testing_images, testing_images))          
#json_string = autoencoder.to_json()

#autoencoder.save_weights('autoencoder_new.h5')

#open('autoencoder_new.h5', 'w').write(json_string)
encoder = Model(inputs = input_img, outputs = hidden_1)

X_train_enc = encoder.predict(training_images)

features_hidden_1 = np.array(X_train_enc)

autoenc_features_hidden_1_wang = pd.DataFrame(features_hidden_1)
autoenc_features_hidden_1_wang
from sklearn.preprocessing import MinMaxScaler

data = autoenc_features_hidden_1_wang

scaler = MinMaxScaler()

scaler.fit(data)

features_norm_wang = scaler.transform(data)

autoenc_normalized_feature_hidden_1_wang = pd.DataFrame(features_norm_wang)
autoenc_normalized_feature_hidden_1_wang
autoenc_normalized_feature_hidden_1_wang.to_csv('autoenc_hidden_1_wang_1.csv',index=False) 