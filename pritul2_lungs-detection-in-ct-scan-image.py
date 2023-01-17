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
import os
os.mkdir("mydata")
os.mkdir("masks")
import zipfile
with zipfile.ZipFile("../input/finding-lungs-in-ct-data/2d_masks.zip", 'r') as zip_ref:
    zip_ref.extractall("./masks")
with zipfile.ZipFile("../input/finding-lungs-in-ct-data/2d_images.zip", 'r') as zip_ref:
    zip_ref.extractall("./mydata")

import cv2
import matplotlib.pyplot as plt
os.chdir("/kaggle/working/mydata")
#print(os.getcwd())
li = os.listdir()
x_data = []
y_data = []
cnt = 0
for i in li:
    img = cv2.imread(i,0)
    #print(img.shape)
    img = cv2.resize(img,(32,32),cv2.INTER_CUBIC)
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    #norm = norm.flatten()
    x_data.append(norm)
    cnt = cnt+1
print(cnt)
input()
os.chdir('..')
os.chdir('masks')
li = os.listdir()
print(os.getcwd())
input()
cnt = 0
for i in li:
    img = cv2.imread(i,0)
    img = cv2.resize(img,(32,32),cv2.INTER_CUBIC)
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    #norm.flatten()
    y_data.append(norm)
    cnt = cnt+1
print(cnt)
import numpy as np
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
print(y_data.shape)
print(x_data.shape)

#CNN accepting ndim=4 #
x_data = x_data[:,:,:,np.newaxis]
print(y_data.shape)
y_data = y_data[:,:,:,np.newaxis]
print(x_data.shape)
print(y_data.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.5)
print(x_train.shape)
print(y_train.shape)
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

input_layer = Input(shape=x_train.shape[1:])
c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
l = Dropout(0.5)(l)
output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
                                                         
model = Model(input_layer, output_layer)
model.summary()
def my_generator(x_train, y_train, batch_size=8):
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=42)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=42)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch
image_batch, mask_batch = next(my_generator(x_train, y_train, 8))
fix, ax = plt.subplots(8,2, figsize=(8,20))
for i in range(8):
    ax[i,0].imshow(image_batch[i,:,:,0])
    ax[i,1].imshow(mask_batch[i,:,:,0])
plt.show()
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
model.build((32,32))
model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
weight_saver = ModelCheckpoint('lung.h5', monitor='val_dice_coef', 
                                              save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
hist = model.fit_generator(my_generator(x_train, y_train, 8),
                           steps_per_epoch = 200,
                           validation_data = (x_test, y_test),
                           epochs=10, verbose=2,
                           callbacks = [weight_saver, annealer])
model.load_weights('lung.h5')



