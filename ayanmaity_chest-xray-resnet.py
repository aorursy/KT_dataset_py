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
path_train = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'
path_test = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test'
path_val = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val'
def img_list(path):
    imglist = []
    labels = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            imglist.append(os.path.join(dirname, filename))
    return imglist
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from skimage.color import rgb2gray

def plot_imgs(path):
    f = plt.figure(figsize=(10,5))
    imglist = img_list(path)
    rand = random.sample(imglist,3)
    for i,img in enumerate(rand):
        imgarr = imread(img)
        print(imgarr.shape)
        if len(imgarr.shape)==3:
            imgarr = rgb2gray(imgarr)
        f.add_subplot(1,3,i+1)
        plt.subplot(131+i)
        plt.imshow(imgarr,cmap='gray')
print('NORMAL')
plot_imgs(path_train+'/NORMAL')
print('PNEUMONIA')
plot_imgs(path_train+'/PNEUMONIA')
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,BatchNormalization,Dropout,Add,Input,Lambda,Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import keras.backend as K
from keras.optimizers import Adam
from keras.applications import VGG16,Xception,ResNet50,MobileNetV2,InceptionV3,VGG19,DenseNet121
##from keras.applications.efficientnet import EfficientNetB2
"""train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        validation_split=0.075,
        horizontal_flip=False)"""
train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.075)
train_generator = train_datagen.flow_from_directory(
        path_train,
        target_size=(224, 224),
        batch_size=128,
        class_mode='binary',
        subset='training')
val_generator = train_datagen.flow_from_directory(
        path_train,
        target_size=(224, 224),
        batch_size=128,
        class_mode='binary',
        subset='validation')
def res_block(inputs,filters,i):
    conv_1 = Conv2D(filters=filters,kernel_size=(3,3),padding='same',activation='relu',name=str('conv_'+str(10*i+1)))(inputs)
    conv_2 = Conv2D(filters=filters,kernel_size=(3,3),padding='same',activation='relu',name=str('conv_'+str(10*i+2)))(conv_1)
    conv_3 = Conv2D(filters=filters,kernel_size=(3,3),padding='same',activation='relu',name=str('conv_'+str(10*i+3)))(conv_2)

    add_1 = Add(name=str('add_'+str(i)))([conv_1,conv_3])
    max_1 = MaxPooling2D(pool_size=(2,2),padding='valid',name=str('max_'+str(i)))(add_1)
    return max_1
input_imgs  = Input((224,224,3))

vgg = DenseNet121(include_top=False, input_shape=(224,224,3), pooling='max')
vgg_last = vgg(input_imgs)

#flat_1 = Flatten(name='flat_3')(vgg_last)
dense_1 = Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='dense_1')(vgg_last)
dense_1 = Dropout(0.25,name='drop_1')(dense_1)
dense_2 = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='dense_2')(dense_1)
dense_2 = Dropout(0.5,name='drop_2')(dense_2)
output = Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01),name='output')(dense_2)

model = Model(inputs=input_imgs,outputs=output,name='CNN Model')
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True,dpi=50)
"""for layer in model.layers:
	layer.trainable = False
model.get_layer('dense_1').trainable = True
model.get_layer('dense_2').trainable = True
model.get_layer('output').trainable = True"""
for layer in model.layers:
    layer.trainable = False
    
    if layer.name.startswith('bn'):
        layer.call(layer.input, training=False)
model.get_layer('dense_1').trainable = True
model.get_layer('dense_2').trainable = True
model.get_layer('output').trainable = True
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(train_generator,validation_data=val_generator,steps_per_epoch=20,epochs=10,verbose=1,callbacks=callbacks_list)
##model.load_weights('weights_best.hdf5')
train_generator1 = train_datagen.flow_from_directory(
        path_train,
        target_size=(224,224),
        batch_size=128,
        class_mode='binary',
        shuffle=False)  # keep data in same order as labels

model.evaluate_generator(train_generator1)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator1 = test_datagen.flow_from_directory(
        path_test,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        shuffle=False)  # keep data in same order as labels

model.evaluate_generator(test_generator1)
model.save('xray_model_xcep.h5')
