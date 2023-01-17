import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from skimage.io import imread,imshow
from skimage.transform import resize
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
#import Augmentor
import cv2
from keras.models import Model,Sequential
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge, Conv2D,Concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.losses import binary_crossentropy
from keras import backend as K
IMG_ROW=IMG_COL=64
IMG_CHANNEL=3
TRAIN_IMG_DIR='../input/train/images/'
TRAIN_MASK_DIR='../input/train/masks/'
def get_img_mask_array(imgpath,maskpath):
    img=imread(imgpath)[:,:,:IMG_CHANNEL]
    img=resize(img,(IMG_ROW,IMG_COL),mode='constant',
               preserve_range=True)
    mask=np.zeros((IMG_ROW,IMG_COL,1),dtype=np.bool)
    mask_=imread(maskpath)
    mask_=np.expand_dims(resize(mask_,(IMG_ROW,IMG_COL),
                                mode='constant',preserve_range=True),
                         axis=-1)
    mask=np.maximum(mask,mask_)
    mask1=[]
    for i in range(len(mask)):
        arr1=[]
        for j in range(len(mask[i])):
            if(mask[i][j]==0.):
                arr1.append(1)
                arr1.append(0)
            else:
                arr1.append(0)
                arr1.append(1)
        mask1.append(np.asarray(arr1))
    mask1=np.asarray(mask1)
    mask=mask1.reshape((IMG_ROW*IMG_COL,2))
    return np.asarray(img),np.asarray(mask)
imgarray=[]
maskarray=[]
for path in os.listdir(TRAIN_IMG_DIR):
    #print(i)
    if(os.path.isfile(TRAIN_IMG_DIR+path)):
        img,mask=get_img_mask_array(TRAIN_IMG_DIR+path,TRAIN_MASK_DIR+path)
        imgarray.append(img)
        maskarray.append(mask)
print(np.asarray(imgarray).shape)
print(np.asarray(maskarray).shape)
def build_model(img_w, img_h, filters):
    n_labels = 2

    kernel = 3

    encoding_layers = [
        Conv2D(64, (kernel, kernel), input_shape=(img_h, img_w, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]

    autoencoder =Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(n_labels, (1, 1), padding='valid', activation="sigmoid"),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Reshape((n_labels, img_h * img_w)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))

    #with open('model_5l.json', 'w') as outfile:
    #    outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))
    autoencoder.summary()
    return autoencoder
model=build_model(IMG_ROW,IMG_COL,10)
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
callbacks=[
    EarlyStopping(patience=5,monitor='val_loss',verbose=1),
    ReduceLROnPlateau(patience=3,monitor='val_loss',verbose=1),
    ModelCheckpoint('model.h5',save_best_only=True)
]
history=model.fit(np.asarray(imgarray),np.asarray(maskarray),epochs=40,
                  validation_split=0.1,callbacks=callbacks)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
predict_result=model.predict(np.asarray(imgarray))
predict_mask=predict_result.reshape((len(predict_result),IMG_ROW,IMG_COL,2))
maskarray=np.asarray(maskarray).reshape((len(maskarray),IMG_ROW,IMG_COL,2))
for i in range(10):
    rnd_id=random.randint(0,len(imgarray)-1)
    f,ax=plt.subplots(1,3,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        if(j==0):
            ax.imshow(imgarray[rnd_id])
        elif(j==1):
            ax.imshow(np.argmax(maskarray[rnd_id],axis=-1))
        else:
            ax.imshow(np.argmax(predict_mask[rnd_id],axis=-1))
        j+=1