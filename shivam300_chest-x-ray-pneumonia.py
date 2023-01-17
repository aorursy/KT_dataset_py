import pandas as pd 

import os

import cv2

import fnmatch

%matplotlib inline

import matplotlib.image as mpimg

import keras

from keras import backend as K

from keras.models import Sequential,load_model,Model

from keras.losses import binary_crossentropy

from keras.optimizers import Adam,RMSprop,Nadam

from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input,SeparableConv2D

from keras.layers.normalization import BatchNormalization

from keras.callbacks  import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

from keras.applications import VGG19,VGG16,ResNet50,Xception

import numpy as np

from keras.utils import np_utils

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
path='../input/chest_xray/chest_xray/train/NORMAL/'

i=0

plt.figure(figsize=(50, 50))

for root,dirs,files in os.walk(path):

    for f in files:

        if(i<6):

            plt.subplot(2, 3, i+1)

            image = cv2.imread(path+f)

            image = cv2.resize(image,(512,512))

            plt.imshow(image)

            i=i+1

        else:

            break

plt.tight_layout()

plt.show()
def test():

    x = []

    y = []

    path = "../input/chest_xray/chest_xray/test/NORMAL/"

    for root,dirs,files in os.walk(path):

        for f in files:

            if fnmatch.fnmatch(f,"*.jpeg"):

                img = cv2.imread(path+f)

                img = cv2.resize(img,(224,224))

                x.append(img)

                y.append(1)

    path = "../input/chest_xray/chest_xray/test/PNEUMONIA/"

    for root,dirs,files in os.walk(path):

        for f in files:

            if fnmatch.fnmatch(f,"*.jpeg"):

                img = cv2.imread(path+f)

                img = cv2.resize(img,(224,224))

                x.append(img)

                y.append(0)

    return np.array(x),np.array(y)
def valid():

    x = []

    y = []

    path = "../input/chest_xray/chest_xray/val/NORMAL/"

    for root,dirs,files in os.walk(path):

        for f in files:

            if fnmatch.fnmatch(f,"*.jpeg"):

                img = cv2.imread(path+f)

                img = cv2.resize(img,(224,224))

                x.append(img)

                y.append(1)

    path = "../input/chest_xray/chest_xray/val/PNEUMONIA/"

    for root,dirs,files in os.walk(path):

        for f in files:

            if fnmatch.fnmatch(f,"*.jpeg"):

                img = cv2.imread(path+f)

                img = cv2.resize(img,(224,224))

                x.append(img)

                y.append(0)

    return np.array(x),np.array(y)
def train():

    x = []

    y = []

    path = "../input/chest_xray/chest_xray/train/NORMAL/"

    for root,dirs,files in os.walk(path):

        for f in files:

            if fnmatch.fnmatch(f,"*.jpeg"):

                img = cv2.imread(path+f)

                img = cv2.resize(img,(224,224))

                x.append(img)

                y.append(1)

    path = "../input/chest_xray/chest_xray/train/PNEUMONIA/"

    for root,dirs,files in os.walk(path):

        for f in files:

            if fnmatch.fnmatch(f,"*.jpeg"):

                img = cv2.imread(path+f)

                img = cv2.resize(img,(224,224))

                x.append(img)

                y.append(0)

    return np.array(x),np.array(y)
valid_x,valid_y = valid()

train_x,train_y = train()
def model():

    image_input = Input(shape=(224,224, 3))

    model=ResNet50(weights='imagenet',include_top=True, input_tensor=image_input)

    last_layer = model.get_layer('fc1000').output

    #out = Dense(1000, activation='relu', name='fc10003')(last_layer)

    out = Dense(500, activation='relu', name='fc10002')(last_layer)

    out = Dense(100, activation='relu',name ='fc1001')(out)

    out = Dense(1, activation='sigmoid',name ='fc1003')(out)

    custom_model = Model(input=image_input,output=out)

    for layer in custom_model.layers[:-5]:

        layer.trainable = False

    

    return custom_model
model1 = model()

model1.summary()
rt=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

weight_path="{}_weights.best.hdf5".format('Chest X-Ray Pneumonia')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model1.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
train_x,train_y = shuffle(train_x,train_y,random_state=0)

valid_x,valid_y = shuffle(valid_x, valid_y,random_state=0)

datagen = ImageDataGenerator(rotation_range=20,zoom_range=0.05,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.05,horizontal_flip=True,vertical_flip=True,fill_mode="nearest")

datagen.fit(train_x)

history=model1.fit_generator(datagen.flow(train_x, train_y, batch_size=32),steps_per_epoch=len(train_x) // 32, epochs=15,validation_data=(train_x,train_y),verbose=1

                             ,callbacks=[rt,checkpoint])
model1.load_weights(weight_path)

model1.save('full_model.h5')
test_x,test_y = test()

pred_Y = model1.predict(test_x, 

                          batch_size = 32, 

                          verbose = True)