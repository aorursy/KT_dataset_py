!rm -r /kaggle/temp
!mkdir /kaggle/temp /kaggle/temp/train /kaggle/temp/test /kaggle/temp/train_labels

!unzip ../input/data-science-bowl-2018/stage1_test.zip -d /kaggle/temp/test

!unzip ../input/data-science-bowl-2018/stage1_train.zip -d /kaggle/temp/train

!unzip ../input/data-science-bowl-2018/stage1_train_labels.csv.zip /kaggle/temp/masks


!ls /kaggle/temp
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
#!/usr/bin/env python

__author__ = "Sreenivas Bhattiprolu"

__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"



# https://www.youtube.com/watch?v=0kiroPnV1tM

# https://www.youtube.com/watch?v=cUHPL_dk17E

# https://www.youtube.com/watch?v=RaswBvMnFxk







"""

@author: Sreenivas Bhattiprolu

"""



import tensorflow as tf

import os

import random

import numpy as np

 

from tqdm import tqdm 



from skimage.io import imread, imshow

from skimage.transform import resize

import matplotlib.pyplot as plt



seed = 42

np.random.seed = seed



IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3



TRAIN_PATH = '/kaggle/temp/train/'

TEST_PATH = '/kaggle/temp/test/'


train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]



X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)



print('Resizing training images and masks')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   

    path = TRAIN_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img  #Fill empty X_train with values from img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  

                                      preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)  

            

    Y_train[n] = mask   



# test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Resizing test images') 

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')



image_x = random.randint(0, len(train_ids))

imshow(X_train[image_x])

plt.show()

imshow(np.squeeze(Y_train[image_x]))

plt.show()











#Build the model

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)



#Contraction path

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)

c1 = tf.keras.layers.Dropout(0.1)(c1)

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)



c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)

c2 = tf.keras.layers.Dropout(0.1)(c2)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

 

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)

c3 = tf.keras.layers.Dropout(0.2)(c3)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

 

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)

c4 = tf.keras.layers.Dropout(0.2)(c4)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

 

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)

c5 = tf.keras.layers.Dropout(0.3)(c5)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)



#Expansive path 

u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

u6 = tf.keras.layers.concatenate([u6, c4])

c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)

c6 = tf.keras.layers.Dropout(0.2)(c6)

c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

 

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

u7 = tf.keras.layers.concatenate([u7, c3])

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)

c7 = tf.keras.layers.Dropout(0.2)(c7)

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

 

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

u8 = tf.keras.layers.concatenate([u8, c2])

c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)

c8 = tf.keras.layers.Dropout(0.1)(c8)

c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

 

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)

c9 = tf.keras.layers.Dropout(0.1)(c9)

c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

 

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

 

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()





model.load_weights("./model.h5")




################################

#Modelcheckpoint

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)



callbacks = [

        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),

        tf.keras.callbacks.TensorBoard(log_dir='logs')]



results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)



####################################




idx = random.randint(0, len(X_train))





preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_test = model.predict(X_test, verbose=1)



 

preds_train_t = (preds_train > 0.5).astype(np.uint8)

preds_val_t = (preds_val > 0.5).astype(np.uint8)

preds_test_t = (preds_test > 0.5).astype(np.uint8)





# Perform a sanity check on some random training samples

ix = random.randint(0, len(preds_train_t))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()

imshow(np.squeeze(preds_train_t[ix]))

plt.show()



# Perform a sanity check on some random validation samples

ix = random.randint(0, len(preds_val_t))

imshow(X_train[int(X_train.shape[0]*0.9):][ix])

plt.show()

imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))

plt.show()

imshow(np.squeeze(preds_val_t[ix]))

plt.show()
model.save_weights("/kaggle/working/model.h5")