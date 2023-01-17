import glob

import cv2

import os

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input, Activation, Flatten

from tensorflow.keras.layers import BatchNormalization,Add,Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate





#validation dataset(20%)

trainPointNames=[]

for dirname, _, filenames in os.walk('/kaggle/input/davis-pointannotation-dataset/Annotations/Point/'):

    a = int(len(filenames)*8/10)

    for i in range(a, len(filenames)):

        trainPointNames.append(os.path.join(dirname, filenames[i]))

            

#valid

validMask = []

validImg = []

validPoint = []



for j in range(len(trainPointNames)):

    validPoint.append(trainPointNames[j])

    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')

    Imgpath = Imgpath.replace('/Point', '/480p')

    Imgpath = Imgpath.replace('.png', '.jpg')

    validImg.append(Imgpath)

    maskpath = trainPointNames[j].replace('/Point', '/480p')

    validMask.append(maskpath)
validsize=300
valid_x = np.zeros((validsize,288,480, 4))

valid_mask_y = np.zeros((validsize,288,480))

    

#valid

for index in range(validsize):

    mask = cv2.imread(validMask[index], cv2.IMREAD_GRAYSCALE)

    dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)

    img = cv2.imread(validImg[index], cv2.IMREAD_COLOR)

    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)

    Point = cv2.imread(validPoint[index], cv2.IMREAD_GRAYSCALE)

    dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)

    vmask = np.array(dstmask)

    vimg = np.array(dstimg)

    valid_Point =np.array(dstPoint)

    valid_x[index, :, :, 0:3] = vimg

    valid_x[index, :, :, 3] = valid_Point

    valid_mask_y[index,:,:] = vmask

valid_x = valid_x.reshape(len(valid_x), 288, 480, 4).astype('float32')/255.0

valid_mask_y = valid_mask_y.reshape(len(valid_mask_y), 288, 480, 1).astype('float32')/255.0
model = load_model('../input/unetv8model/UNetv8.h5')

fig, ax = plt.subplots(len(valid_x), 3, figsize=(10,700))

preds = model.predict(valid_x)



for i, pred in enumerate(preds):

    ax[i, 0].imshow(valid_x[i, :, :, 0:3].squeeze())

    ax[i, 1].imshow(valid_mask_y[i].squeeze(), cmap='gray')

    ax[i, 2].imshow(pred.squeeze(), cmap='gray')
model = load_model('../input/unetv6model/UNetv6.h5')

fig, ax = plt.subplots(len(valid_x), 3, figsize=(10,700))

preds = model.predict(valid_x)



for i, pred in enumerate(preds):

    ax[i, 0].imshow(valid_x[i, :, :, 0:3].squeeze())

    ax[i, 1].imshow(valid_mask_y[i].squeeze(), cmap='gray')

    ax[i, 2].imshow(pred.squeeze(), cmap='gray')