# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils

# Helper libraries

import numpy as np

import matplotlib.pyplot as plt

import glob

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.applications import imagenet_utils

print(tf.__version__)



import random

import os



# Any results you write to the current directory are saved as output.
! git clone https://github.com/tuenv2kProPTIT/AIF-HOME-WORK-PROTPTIT.git
IMG_WIDTH=64

IMG_HEIGHT=64

link_to_data_folder='/kaggle/working/AIF-HOME-WORK-PROTPTIT/contest-2/data/contest-2/final_train/final_train'

link_to_test_folder='/kaggle/working/AIF-HOME-WORK-PROTPTIT/contest-2/data/contest-2/final_test'

max_data=7169

max_test=1000
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

training_data = train_datagen.flow_from_directory(link_to_data_folder,

      target_size = (64, 64),

      batch_size = max_data,

      class_mode = 'categorical')

test_data=test_datagen.flow_from_directory(link_to_test_folder,

      target_size = (64, 64),

      batch_size = max_test,

      shuffle=False)

(x_train,y_train)=training_data[0]

(x_test,file_name_test)=test_data[0][0],test_data.filenames
import pandas as pd

data = pd.read_csv("/kaggle/working/AIF-HOME-WORK-PROTPTIT/contest-2/solve (5).csv")

y_test=data['class'].values.reshape(-1,1)

# print(y_test)

for i in range(1000):

  if y_test[i][0]==33:

    y_test[i][0]=5

    continue



  if y_test[i][0]==0:

    y_test[i][0]=0

    continue

  if y_test[i][0]==10:

      y_test[i][0]=1

      continue



  if y_test[i][0]==14:

      y_test[i][0]=2

      continue

  if y_test[i][0]==2:

    y_test[i][0]=3

    continue

  if y_test[i][0]==22:

    y_test[i][0]=4

    continue

  if y_test[i][0]==34:

    y_test[i][0]=6

    continue

  y_test[i][0]=7

np.random.seed(24+11+2000) # :V

y_test= np_utils.to_categorical(y_test,num_classes=8,dtype='int')

label=np.random.choice(x_train.shape[0], 10, replace=False)

X=x_train[label]

Y=y_train[label]

fig,ax=plt.subplots(5,2,figsize=(12,12))

for i in range(5):

    ax[i,0].imshow(X[i*2])

    ax[i,0].set_title('label train is : {}'.format(np.argmax(Y[i*2])))

    ax[i,1].imshow(X[i*2+1])

    ax[i,1].set_title('label train is : {}'.format(np.argmax(Y[i*2+1])))

fig.tight_layout(pad=3.0)



model =tf.keras.Sequential(

    [

      tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(64,64,3),padding='same',activation='relu'),

      tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),

      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

      tf.keras.layers.Dropout(0.3),

      tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),

      tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),

      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

      tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),

      tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),

      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(512),

      tf.keras.layers.BatchNormalization(center=True, scale=False),

      tf.keras.layers.Activation('relu'),

      tf.keras.layers.Dropout(0.3),

      tf.keras.layers.Dense(8, activation='softmax')

    ]

)
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("/kaggle/working/AIF-HOME-WORK-PROTPTIT/contest-2/my_keras_model.h5",

        save_best_only=True)



model.summary()

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)



testgen=ImageDataGenerator()
H = model.fit(datagen.flow(x_train, y_train, batch_size=32), 

                        validation_data=(x_test, y_test),

                        epochs=15,callbacks=[checkpoint_cb])


path_index_test=[p.split(os.path.sep)[-1] for p in file_name_test]







XX=[0,10,14,2,22,33,34,6]

model = tf.keras.models.load_model("/kaggle/working/AIF-HOME-WORK-PROTPTIT/contest-2/my_keras_model.h5")

test_predict=np.argmax(model.predict(x_test),axis=1)

for i in range(1000):

    test_predict[i]=XX[test_predict[i]]

output = pd.DataFrame({'class': test_predict, 'path':path_index_test})

output.to_csv('my_submission1.csv', index=False)

print("Your submission was successfully saved!")