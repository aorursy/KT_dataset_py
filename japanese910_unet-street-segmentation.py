import os

import sys

import cv2

import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

from keras.layers.merge import concatenate

from keras.models import Model, load_model

from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt

from keras import backend as K

import tensorflow as tf

%matplotlib inline
train_image_path = sorted(os.listdir("../input/street/train/train"))

train_label_path = sorted(os.listdir("../input/street/train_labels/train_labels"))



val_image_path = sorted(os.listdir("../input/street/val/val"))

val_label_path = sorted(os.listdir("../input/street/val_labels/val_labels"))



test_image_path = sorted(os.listdir("../input/street/test/test"))

test_label_path = sorted(os.listdir("../input/street/test_labels/test_labels"))
def load_images(inputdir, inputpath, imagesize):

    imglist = []

    

    for i in range(len(inputpath)):

        img = cv2.imread(inputdir+inputpath[i], cv2.IMREAD_COLOR) 

        img = cv2.resize(img, (imagesize, imagesize), interpolation = cv2.INTER_AREA)

        #img = img[::-1] 

        imglist.append(img)

        

    return imglist
IMAGE_SIZE = 128



train_image = load_images("../input/street/train/train/", train_image_path, IMAGE_SIZE)

train_label = load_images("../input/street/train_labels/train_labels/", train_label_path, IMAGE_SIZE)



val_image = load_images("../input/street/val/val/", val_image_path, IMAGE_SIZE)

val_label = load_images("../input/street/val_labels/val_labels/", val_label_path, IMAGE_SIZE)



test_image = load_images("../input/street/test/test/", test_image_path, IMAGE_SIZE)

test_label = load_images("../input/street/test_labels/test_labels/", test_label_path, IMAGE_SIZE)





train_image /= np.max(train_image)

train_label /= np.max(train_label)



val_image /= np.max(val_image)

val_label /= np.max(val_label)



test_image /= np.max(test_image)

test_label /= np.max(test_label)
num = 64

plt.figure(figsize=(14, 7))



ax = plt.subplot(1, 2, 1)

plt.imshow(np.squeeze(train_image[num]))



ax = plt.subplot(1, 2, 2)

plt.imshow(np.squeeze(train_label[num]))
def Unet():

    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))



    enc1 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(input_img)

    enc1 = BatchNormalization()(enc1)

    enc1 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(enc1)

    enc1 = BatchNormalization()(enc1)

    down1 = MaxPooling2D(pool_size=2, strides=2)(enc1)

    

    enc2 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(down1)

    enc2 = BatchNormalization()(enc2)

    enc2 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(enc2)

    enc2 = BatchNormalization()(enc2)

    down2 = MaxPooling2D(pool_size=2, strides=2)(enc2)



    enc3 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(down2)

    enc3 = BatchNormalization()(enc3)

    enc3 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(enc3)

    enc3 = BatchNormalization()(enc3)

    down3 = MaxPooling2D(pool_size=2, strides=2)(enc3)

    

    enc4 = Conv2D(1024, kernel_size=3, strides=1, activation="relu", padding="same")(down3)

    enc4 = BatchNormalization()(enc4)

    enc4 = Conv2D(1024, kernel_size=3, strides=1, activation="relu", padding="same")(enc4)

    enc4 = BatchNormalization()(enc4)

    down4 = MaxPooling2D(pool_size=2, strides=2)(enc4)

    

    enc5 = Conv2D(2048, kernel_size=3, strides=1, activation="relu", padding="same")(down4)

    enc5 = BatchNormalization()(enc5)

    enc5 = Conv2D(2048, kernel_size=3, strides=1, activation="relu", padding="same")(enc5)

    enc5 = BatchNormalization()(enc5)



    up4 = UpSampling2D(size=2)(enc5)

    dec4 = concatenate([up4, enc4], axis=-1)

    dec4 = Conv2D(1024, kernel_size=3, strides=1, activation="relu", padding="same")(dec4)

    dec4 = BatchNormalization()(dec4)

    dec4 = Conv2D(1024, kernel_size=3, strides=1, activation="relu", padding="same")(dec4)

    dec4 = BatchNormalization()(dec4)

    

    up3 = UpSampling2D(size=2)(dec4)

    dec3 = concatenate([up3, enc3], axis=-1)

    dec3 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(dec3)

    dec3 = BatchNormalization()(dec3)

    dec3 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(dec3)

    dec3 = BatchNormalization()(dec3)



    up2 = UpSampling2D(size=2)(dec3)

    dec2 = concatenate([up2, enc2], axis=-1)

    dec2 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)

    dec2 = BatchNormalization()(dec2)

    dec2 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)

    dec2 = BatchNormalization()(dec2)

    

    up1 = UpSampling2D(size=2)(dec2)

    dec1 = concatenate([up1, enc1], axis=-1)

    dec1 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)

    dec1 = BatchNormalization()(dec1)

    dec1 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)

    dec1 = BatchNormalization()(dec1)

    

    dec1 = Conv2D(3, kernel_size=1, strides=1, activation="sigmoid", padding="same")(dec1)

    

    model = Model(input=input_img, output=dec1)

    

    return model



model = Unet()

model.summary()
def castF(x):

    return K.cast(x, K.floatx())



def castB(x):

    return K.cast(x, bool)



def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative

    intersection = true * pred

    notTrue = 1 - true

    union = true + (notTrue * pred)



    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())



def competitionMetric2(true, pred): #any shape can go - can't be a loss function



    tresholds = [0.5 + (i*.05)  for i in range(10)]



    #flattened images (batch, pixels)

    true = K.batch_flatten(true)

    pred = K.batch_flatten(pred)

    pred = castF(K.greater(pred, 0.5))



    #total white pixels - (batch,)

    trueSum = K.sum(true, axis=-1)

    predSum = K.sum(pred, axis=-1)



    #has mask or not per image - (batch,)

    true1 = castF(K.greater(trueSum, 1))    

    pred1 = castF(K.greater(predSum, 1))



    #to get images that have mask in both true and pred

    truePositiveMask = castB(true1 * pred1)



    #separating only the possible true positives to check iou

    testTrue = tf.boolean_mask(true, truePositiveMask)

    testPred = tf.boolean_mask(pred, truePositiveMask)



    #getting iou and threshold comparisons

    iou = iou_loss_core(testTrue,testPred) 

    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]



    #mean of thressholds for true positives and total sum

    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)

    truePositives = K.sum(truePositives)



    #to get images that don't have mask in both true and pred

    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1

    trueNegatives = K.sum(trueNegatives) 



    return (truePositives + trueNegatives) / castF(K.shape(true)[0])
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[competitionMetric2])



initial_learningrate = 0.002



def lr_decay(epoch):

    if epoch < 30:

        return initial_learningrate

    else:

        return initial_learningrate * 0.99 ** epoch



training = model.fit(train_image, train_label, epochs=50, batch_size=32, shuffle=True, validation_data=(val_image, val_label), verbose=1,callbacks=[LearningRateScheduler(lr_decay,verbose=1)])
results = model.predict(test_image,verbose=1)
import random

n = 20

Num = 10

plt.figure(figsize=(140, 14))



for i in range(3):

    

   # 原画像

    ax = plt.subplot(2, n, i+1)

    plt.imshow(test_image[Num])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

   

    # 推定結果画像

    ax = plt.subplot(2, n, i+1+n)

    plt.imshow(results[Num])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    Num += 20

    

plt.show()
