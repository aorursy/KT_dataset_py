%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import random

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import tensorflow as tf

import time



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D

from keras import optimizers

from keras.layers.normalization import BatchNormalization as BN



from keras.layers import Lambda, Reshape, Add, AveragePooling2D, MaxPooling2D, Concatenate, SeparableConv2D

from keras.models import Model

from keras.losses import mse, binary_crossentropy

from keras.utils import plot_model

from keras import backend as K



from keras.callbacks import ModelCheckpoint



from keras.regularizers import l2



from keras.preprocessing.image import array_to_img, img_to_array, load_img



from sklearn.model_selection import train_test_split



from PIL import Image, ImageDraw, ImageFilter

print(os.listdir("../input"))
def resnet_en(data, filters, kernel_size, dilation_rate,option=False):

    if option:

        x=BN()(data)

        x = Activation("relu")(x)

        x=Conv2D(filters=filters,kernel_size=kernel_size,dilation_rate=dilation_rate,strides=(1,1),padding="same")(x)



        x=BN()(x)

        x = Activation("relu")(x)

        x=Conv2D(filters=filters,kernel_size=kernel_size,dilation_rate=(1,1),strides=(2,2),padding="same")(x)

        

    else:

        x=BN()(data)

        x = Activation("relu")(x)

        x=Conv2D(filters=filters,kernel_size=kernel_size,dilation_rate=dilation_rate,strides=(1,1),padding="same")(x)



        x=BN()(x)

        x = Activation("relu")(x)

        x=Conv2D(filters=filters,kernel_size=kernel_size,dilation_rate=dilation_rate,strides=(1,1),padding="same")(x)



    return x
def shortcut_en(x, residual):

    '''shortcut connection を作成する。

    '''

    x_shape = K.int_shape(x)

    residual_shape = K.int_shape(residual)



    if x_shape == residual_shape:

        # x と residual の形状が同じ場合、なにもしない。

        shortcut = x

    else:

        # x と residual の形状が異なる場合、線形変換を行い、形状を一致させる。

        stride_w = int(round(x_shape[1] / residual_shape[1]))

        stride_h = int(round(x_shape[2] / residual_shape[2]))



        shortcut = Conv2D(filters=residual_shape[3],

                          kernel_size=(1, 1),

                          strides=(stride_w, stride_h),

                          kernel_initializer='he_normal',

                          kernel_regularizer=l2(1.e-4))(x)

    return Add()([shortcut, residual])
def upsampling_unit(x, first_filter, number):

    for i in range(number):

        x = UpSampling2D((2,2))(x)

        #16*100

        x=Conv2D(filters=first_filter//(2**i),kernel_size=(3,3),strides=(1,1),padding="same")(x)

        x = BN()(x)

        x = Activation("relu")(x)



    return x
def PSP_unit(x, filters):

    x1 = MaxPooling2D((1,1),padding="same")(x)

    x1=Conv2DTranspose(filters=filters//4,kernel_size=(1,1),strides=(1,1),padding="same")(x1)

    x1 = BN()(x1)

    x1 = Activation("relu")(x1)



    x2 = MaxPooling2D((2,2),padding="same")(x)

    x2=Conv2DTranspose(filters=filters//4,kernel_size=(1,1),strides=(2,2),padding="same")(x2)

    x2 = BN()(x2)

    x2 = Activation("relu")(x2)



    x3 = MaxPooling2D((4,5),padding="same")(x)

    x3=Conv2DTranspose(filters=filters//4,kernel_size=(1,1),strides=(4,5),padding="same")(x3)

    x3 = BN()(x3)

    x3 = Activation("relu")(x3)



    x4 = MaxPooling2D((8,10),padding="same")(x)

    x4=Conv2DTranspose(filters=filters//4,kernel_size=(1,1),strides=(8,10),padding="same")(x4)

    x4 = BN()(x4)

    x4 = Activation("relu")(x4)



    return Concatenate()([x,x1,x2,x3,x4])
inputs = Input(shape=(img_height, img_width, 1))



#128*800

fx = resnet_en(inputs, 16, (3,3), (1,1))

x = shortcut_en(inputs, fx)

fx = resnet_en(x, 16, (3,3), (1,1), True)

x = shortcut_en(x, fx)



#64*400

fx = resnet_en(x, 32, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 32, (3,3), (1,1), True)

x = shortcut_en(x, fx)



#32*200

fx = resnet_en(x, 64, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 64, (3,3), (1,1))

x = shortcut_en(x, fx)



fx = resnet_en(x, 64, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 64, (3,3), (1,1), True)

x = shortcut_en(x, fx)



#16*100

fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)



fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)



fx = resnet_en(x, 256, (3,3), (2,2))

x = shortcut_en(x, fx)

fx = resnet_en(x, 256, (3,3), (2,2))

x = shortcut_en(x, fx)



fx = resnet_en(x, 256, (3,3), (2,2))

x = shortcut_en(x, fx)

fx = resnet_en(x, 256, (3,3), (2,2))

x = shortcut_en(x, fx)



fx = resnet_en(x, 256, (3,3), (4,4))

x = shortcut_en(x, fx)

fx = resnet_en(x, 256, (3,3), (4,4))

x = shortcut_en(x, fx)



fx = resnet_en(x, 256, (3,3), (4,4))

x = shortcut_en(x, fx)

fx = resnet_en(x, 256, (3,3), (4,4))

x = shortcut_en(x, fx)



x=Conv2D(filters=512,kernel_size=(3,3),dilation_rate=(2,2),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x=Conv2D(filters=512,kernel_size=(3,3),dilation_rate=(2,2),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x=Conv2D(filters=512,kernel_size=(3,3),dilation_rate=(1,1),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x=Conv2D(filters=512,kernel_size=(3,3),dilation_rate=(1,1),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = PSP_unit(x, 512)



x1 = Conv2DTranspose(filters=256,kernel_size=(1,1),strides=(2,2),padding="same",kernel_initializer='he_normal',

                          kernel_regularizer=l2(1.e-4))(x)



x = UpSampling2D((2,2))(x)

x=Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = Add()([x,x1])



x2 = Conv2DTranspose(filters=128,kernel_size=(1,1),strides=(2,2),padding="same",kernel_initializer='he_normal',

                          kernel_regularizer=l2(1.e-4))(x)



x = UpSampling2D((2,2))(x)

x=Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = Add()([x,x2])



x3 = Conv2DTranspose(filters=64,kernel_size=(1,1),strides=(2,2),padding="same",kernel_initializer='he_normal',

                          kernel_regularizer=l2(1.e-4))(x)



x = UpSampling2D((2,2))(x)

x=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = Add()([x,x3])



x4 = Conv2DTranspose(filters=8,kernel_size=(1,1),strides=(2,2),padding="same",kernel_initializer='he_normal',

                          kernel_regularizer=l2(1.e-4))(x)



x = UpSampling2D((2,2))(x)

x=Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = Add()([x,x4])



inputs_2 = Input(shape=(256, 1600, 1))

xa=Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding="same")(inputs_2)

xa = BN()(xa)

xa = Activation("relu")(xa)



x = Add()([x,xa])



x=Conv2D(filters=5,kernel_size=(1,1),strides=(1,1),padding="same")(x)

outputs = Activation('softmax')(x)



# instantiate decoder model

model = Model([inputs, inputs_2], outputs)

model.summary()



model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),

             loss="categorical_crossentropy", metrics=["accuracy"])
def data_generator(train_pd, img_index, batch_size, abs_path, img_width, img_height, data_aug):

    '''data generator for fit_generator'''

    n = len(img_index)

    i = 0

    while True:

        image_data_resize = []

        image_data = []

        fmap_data = []

        for b in range(batch_size):

            if i==0:

                np.random.shuffle(img_index)

            image_resize, image, fmap = get_random_data(train_pd, img_index[i], abs_path, img_width, img_height, data_aug)

            image_data_resize.append(image_resize)

            image_data.append(image)

            fmap_data.append(fmap)

            i = (i+1) % n

        image_data_resize = np.array(image_data_resize)

        image_data = np.array(image_data)

        fmap_data = np.array(fmap_data)

        yield [image_data_resize, image_data], fmap_data



def data_generator_wrapper(train_pd, img_index, batch_size, abs_path, img_width, img_height, data_aug):

    n = len(img_index)

    if n==0 or batch_size<=0: return None

    return data_generator(train_pd, img_index, batch_size, abs_path, img_width, img_height, data_aug)
img_width, img_height = 800, 128

batch_size = 8
model.load_weights("../input/monika/best_weight.h5")
def make_testdata(a):



    data = []

    c = 1



    for i in range(a.shape[0]-1):

        if a[i]+1 == a[i+1]:

            c += 1

            if i == a.shape[0]-2:

                data.append(str(a[i-c+2]))

                data.append(str(c))



        if a[i]+1 != a[i+1]:

            data.append(str(a[i-c+1]))

            data.append(str(c))

            c = 1



    data = " ".join(data)

    return data
start = time.time()



test_path = "../input/severstal-steel-defect-detection/test_images/"



test_list = os.listdir(test_path)



data = []



for fn in test_list:

    abs_name = test_path + fn

    seed_image = cv2.imread(abs_name)

    seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

    seed_image_resize = cv2.resize(seed_image, dsize=(img_width, img_height))

    seed_image_resize = np.expand_dims(seed_image_resize, axis=-1)

    seed_image_resize = np.expand_dims(seed_image_resize, axis=0)

    seed_image_resize = seed_image_resize/255



    seed_image = np.expand_dims(seed_image, axis=-1)

    seed_image = np.expand_dims(seed_image, axis=0)

    seed_image = seed_image/255



    pred = model.predict([seed_image_resize, seed_image])[0]

    

    for i in range(4):

        

        pred_fi = pred[:,:,i+1].T.flatten()

        pred_fi = np.where(pred_fi > 0.5, 1, 0)

        pred_fi_id = np.where(pred_fi == 1)

        pred_fi_id = make_testdata(pred_fi_id[0])

        x = [fn + "_" + str(i+1), pred_fi_id]

        data.append(x)



elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
columns = ['ImageId_ClassId', 'EncodedPixels']

d = pd.DataFrame(data=data, columns=columns, dtype='str')

d.to_csv("submission.csv",index=False)

df = pd.read_csv("submission.csv")
from IPython.display import FileLink

FileLink(r"submission.csv")