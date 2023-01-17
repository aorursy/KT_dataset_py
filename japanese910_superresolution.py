import cv2

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import load_model

from sklearn.model_selection import train_test_split



from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add

from keras.models import Model

import keras.backend as K

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, ModelCheckpoint



%matplotlib inline
def load_images(inputdir, inputpath, imagesize):

    imglist = []

    

    for i in range(len(inputpath)):

        img = cv2.imread(inputdir+inputpath[i], cv2.IMREAD_COLOR) 

        img = img[::-1] 

        imglist.append(img)

        

    return imglist
IMAGE_SIZE = 128





image_path = sorted(os.listdir("../input/super-resolution/flower_images_scale3/flower_images_scale3"))

label_path = sorted(os.listdir("../input/super-resolution/flower_images_scale6/flower_images_scale6"))





image =load_images("../input/super-resolution/flower_images_scale3/flower_images_scale3/", image_path, IMAGE_SIZE)

label =load_images("../input/super-resolution/flower_images_scale6/flower_images_scale6/", label_path, IMAGE_SIZE)

    

image /= np.max(image)

label /= np.max(label)



image.shape, label.shape
num = 20



plt.figure(figsize=(14, 7))



ax = plt.subplot(1, 2, 1)

plt.imshow(np.squeeze(image[num]))



ax = plt.subplot(1, 2, 2)

plt.imshow(np.squeeze(label[num]))


def network_ddsrcnn():

    input_img = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3))

    

    enc1 = Conv2D(64,kernel_size=3,activation="relu",padding="same")(input_img)

    enc1 = Conv2D(64,kernel_size=3,activation="relu",padding="same")(enc1)

    down1 = MaxPooling2D(pool_size=2)(enc1)



    enc2 = Conv2D(128,kernel_size=3,activation="relu",padding="same")(down1)

    enc2 = Conv2D(128,kernel_size=3,activation="relu",padding="same")(enc2)

    down2 = MaxPooling2D(pool_size=2)(enc2)

    

    enc3 = Conv2D(256,kernel_size=3,activation="relu",padding="same")(down2)

    

    up3 = UpSampling2D(size=2)(enc3)

    dec3 = Conv2D(128,kernel_size=3,activation="relu",padding="same")(up3)

    dec3 = Conv2D(128,kernel_size=3,activation="relu",padding="same")(dec3)

    

    add2 = Add()([dec3,enc2])

    up2 = UpSampling2D(size=2)(add2)

    dec2 = Conv2D(64,kernel_size=3,activation="relu",padding="same")(up2)

    dec2 = Conv2D(64,kernel_size=3,activation="relu",padding="same")(dec2)

    

    add1 = Add()([dec2,enc1])

    dec1 = Conv2D(3,kernel_size=5,activation="linear",padding="same")(add1)

    

    model = Model(input_img,dec1)

    return model



model = network_ddsrcnn()

model.summary()
initial_learningrate=2e-3

    

def lr_decay(epoch):

    if epoch < 500:

        return initial_learningrate

    else:

        return initial_learningrate * 0.99 ** epoch





def psnr(y_true,y_pred):

    return -10*K.log(K.mean(K.flatten((y_true-y_pred))**2))/np.log(10)



model.compile(loss="mean_squared_error",optimizer=Adam(lr=initial_learningrate),metrics=[psnr])

model.fit(label,image,epochs=500,batch_size=32,verbose=1,callbacks=[LearningRateScheduler(lr_decay,verbose=1)])
results = model.predict(image,verbose=1)
n = 8



plt.figure(figsize=(14, 7))



ax = plt.subplot(1, 3, 1)

plt.imshow(np.squeeze(results[n]))



ax = plt.subplot(1, 3, 2)

plt.imshow(np.squeeze(image[n]))



ax = plt.subplot(1, 3, 3)

plt.imshow(np.squeeze(label[n]))