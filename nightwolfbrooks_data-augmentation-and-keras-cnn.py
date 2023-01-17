from __future__ import print_function

import numpy as np # For numerical fast numerical calculations

import matplotlib.pyplot as plt # For making plots

import pandas as pd # Deals with data

import seaborn as sns # Makes beautiful plots

import keras 

import sys 

from pandas import pandas as pd

import category_encoders as ce

import datetime

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

import os

import glob

import numpy as np

import scipy as sp

import pandas as pd

# skimage

from skimage.io import imshow, imread, imsave

from skimage.transform import rotate, AffineTransform, warp,rescale, resize, downscale_local_mean

from skimage import color,data

from skimage.exposure import adjust_gamma

from skimage.util import random_noise

# imgaug

import imageio

import imgaug as ia

import imgaug.augmenters as iaa

# Albumentations

import albumentations as A 

# Keras

from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 

#visualisation

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

#%matplotlib inline

import seaborn as sns

from IPython.display import HTML, Image

import cv2

import os

import matplotlib.ticker as ticker

import matplotlib.pyplot as plt

import seaborn as sns



# load data



p_train=pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')

p_test=pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
p_train=pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')

p_test=pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')



target = p_train[['healthy', 'multiple_diseases', 'rust', 'scab']]

test_ids = p_test['image_id']



img_size=224
colors=['#78C850','#6890F0']

p_train['healthy'].value_counts().plot(kind='bar',title='Healthy Frequency Count',color=colors)
p_train['multiple_diseases'].value_counts().plot(kind='bar',title='Multi-Diseases Frequency Count',color=colors)
p_train['rust'].value_counts().plot(kind='bar',title='Rust Frequency Count',color=colors)
p_train['scab'].value_counts().plot(kind='bar',title='Scab Frequency Count',color=colors)
train_image=[]

for name in p_train['image_id']:

    path='../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    train_image.append(image)



fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(train_image[i])

    

    

test_image=[]

for name in p_test['image_id']:

    path='../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    test_image.append(image)

fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(test_image[i])    



from keras.preprocessing.image import img_to_array



x_train = np.ndarray(shape=(len(train_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in train_image:

    x_train[i]=img_to_array(image)

    x_train[i]=train_image[i]

    i=i+1

x_train=x_train/255

print('Train Shape: {}'.format(x_train.shape))



x_test = np.ndarray(shape=(len(test_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in test_image:

    x_test[i]=img_to_array(image)

    x_test[i]=test_image[i]

    i=i+1

    

x_test=x_test/255

print('Test Shape: {}'.format(x_test.shape))
y = p_train.copy()

del y['image_id']

y.head()



y_train = np.array(y.values)

print(y_train.shape,y_train[0])



from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=115)



x_train.shape, x_val.shape, y_train.shape, y_val.shape
from imblearn.over_sampling import SMOTE 



sm = SMOTE(random_state = 115) 

 

x_train, y_train = sm.fit_resample(x_train.reshape((-1, img_size * img_size * 3)), y_train)

x_train = x_train.reshape((-1, img_size, img_size, 3))

x_train.shape, y_train.sum(axis=0)
from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping

LR_reduce=ReduceLROnPlateau(monitor='val_accuracy',

                            factor=.5,

                            patience=10,

                            min_lr=.000001,

                            verbose=1)



ES_monitor=EarlyStopping(monitor='val_loss',

                          patience=20)



reg = .0005
from keras.models import Model, Sequential, load_model, Input

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.utils import plot_model

from keras.regularizers import l2

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten

from keras.layers import GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.applications.resnet50 import ResNet50

from keras.models import Model

import keras

from keras import optimizers



model = Sequential()



model.add(Conv2D(32, kernel_size=(5,5),activation='relu', input_shape=(img_size, img_size, 3), kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(Conv2D(128, kernel_size=(5,5),activation='relu', kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))

model.add(Dropout(.25))



model.add(Conv2D(32, kernel_size=(3,3),activation='relu', kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(Conv2D(128, kernel_size=(3,3),activation='relu',kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))

model.add(Dropout(.25))





model.add(Conv2D(128, kernel_size=(5,5),activation='relu', kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(Conv2D(512, kernel_size=(5,5),activation='relu',kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))

model.add(Dropout(.25))



model.add(Conv2D(128, kernel_size=(3,3),activation='relu',kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(Conv2D(512, kernel_size=(3,3),activation='relu',kernel_regularizer=l2(reg)))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))

model.add(Dropout(.25))



model.add(Flatten())

model.add(Dense(300,activation='relu'))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(Dropout(.25))

model.add(Dense(200,activation='relu'))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(Dropout(.25))

model.add(Dense(100,activation='relu'))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(Dropout(.25))

model.add(Dense(4,activation='softmax'))



model.summary()



from keras.preprocessing.image import ImageDataGenerator



model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy']

              )
datagen = ImageDataGenerator(rotation_range=45,

                             shear_range=.25,

                              zoom_range=.25,

                              width_shift_range=.25,

                              height_shift_range=.25,

                              rescale=1/255,

                              brightness_range=[.5,1.5],

                              horizontal_flip=True,

                              vertical_flip=True,

                              fill_mode='nearest'

#                              featurewise_center=True,

#                              samplewise_center=True,

#                              featurewise_std_normalization=True,

#                              samplewise_std_normalization=True,

#                              zca_whitening=True

                              )



history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=24),

                              epochs=300,

                              steps_per_epoch=x_train.shape[0] // 24,

                              verbose=1,

                              callbacks=[ES_monitor,LR_reduce],

                              validation_data=datagen.flow(x_val, y_val,batch_size=24),

                              validation_steps=x_val.shape[0]//24

                              )
from matplotlib import pyplot as plt



h = history.history



offset = 5

epochs = range(offset, len(h['loss']))



plt.figure(1, figsize=(20, 6))



plt.subplot(121)

plt.xlabel('epochs')

plt.ylabel('loss')

plt.plot(epochs, h['loss'][offset:], label='train')

plt.plot(epochs, h['val_loss'][offset:], label='val')

plt.legend()



plt.subplot(122)

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.plot(h[f'accuracy'], label='train')

plt.plot(h[f'val_accuracy'], label='val')

plt.legend()



plt.show()



from sklearn.metrics import roc_auc_score



pred_test = model.predict(x_val)

roc_sum = 0

for i in range(4):

    score = roc_auc_score(y_val[:, i], pred_test[:, i])

    roc_sum += score

    print(f'{score:.3f}')



roc_sum /= 4

print(f'totally:{roc_sum:.3f}')
pred = model.predict(x_test)



res = pd.DataFrame()

res['image_id'] = test_ids

res['healthy'] = pred[:, 0]

res['multiple_diseases'] = pred[:, 1]

res['rust'] = pred[:, 2]

res['scab'] = pred[:, 3]

res.to_csv('submission.csv', index=False)

res.head(10)