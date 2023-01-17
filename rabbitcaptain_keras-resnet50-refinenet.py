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



from keras.applications.resnet50 import ResNet50

from keras.applications.xception import Xception

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D

from keras import optimizers

from keras.layers.normalization import BatchNormalization as BN



from keras.layers import Lambda, Reshape, Add, AveragePooling2D, MaxPooling2D, Concatenate, SeparableConv2D, ZeroPadding2D

from keras.models import Model

from keras.losses import mse, binary_crossentropy

from keras.utils import plot_model

from keras import backend as K



from keras.regularizers import l2



from keras.callbacks import ModelCheckpoint



from keras.preprocessing.image import array_to_img, img_to_array, load_img



from sklearn.model_selection import train_test_split



from PIL import Image, ImageDraw, ImageFilter

print(os.listdir("../input/keras-pretrained-models"))
train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
train['ImageId'] = train['ImageId_ClassId'].str[:-2]

train['ClassId'] = train['ImageId_ClassId'].str[-1:]

train = train[['ImageId','ClassId','EncodedPixels']]

train
train = train.fillna(0)
train
start = time.time()



filelist = os.listdir("../input/severstal-steel-defect-detection/train_images/")



train_img = []



for i in filelist:

    x = train[train["ImageId"] == i]

    if len(x[x["EncodedPixels"] == 0]) == 4:

        pass

        

    else:

        train_img.append(i)

        

train_img = np.array(train_img)



elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
train_img
img_name = train["ImageId"][43212]

img_name
abs_path = "../input/severstal-steel-defect-detection/train_images/"
seed_image = cv2.imread(abs_path+img_name)

plt.figure(figsize=(15,15))

plt.imshow(seed_image)
df_exact = train[train["ImageId"] == img_name]

df_exact
df_exact2 = df_exact[df_exact["ClassId"] == "1"]

df_exact2
segment_4 = []

for i in range(4):

    x = train[train["ImageId"] == img_name]

    x2 = x[x["ClassId"] == str(i+1)]

    x3 = x2["EncodedPixels"].values[0]

    

    if x3 ==0:

        x4 = "ok"

        

    else:

        x4 = x3.split()

        

    segment_4.append(x4)



segment_4 = np.array(segment_4)
segment_4[3]
#セグメンテーションの生成

seg_img = np.zeros([seed_image.shape[0], seed_image.shape[1],4], dtype=np.uint8)



for j in range(4):

    

    seg_np = np.zeros([seed_image.shape[0]*seed_image.shape[1]], dtype=np.uint8)

    

    if segment_4[j]=="ok":

        pass

    

    else:

        for i in range(len(segment_4[j])//2):

            start = int(segment_4[j][2*i])

            length = int(segment_4[j][2*i+1])

            seg_np[start:start+length]=1



    seg_img[:,:,j] = seg_np.reshape([seed_image.shape[1],seed_image.shape[0]]).T
seed_image = cv2.resize(seed_image, dsize=(800, 128))

seg_img = cv2.resize(seg_img, dsize=(800, 128))
plt.figure(figsize=(15,15))

plt.imshow(seed_image)
plt.figure(figsize=(15,15))

plt.imshow(seg_img[:,:,0],"gray",vmin=0,vmax=1)
def vertical_flip(image,fmap, rate=0.5):

    if np.random.rand() < rate:

        image = image[::-1, :, :]

        fmap = fmap[::-1, :, :]

    return image, fmap





def horizontal_flip(image,fmap, rate=0.5):

    if np.random.rand() < rate:

        image = image[:, ::-1, :]

        fmap = fmap[:, ::-1, :]

    return image, fmap



def image_translation(img,fmap):

    params = np.random.randint(-50, 51)

    if not isinstance(params, list):

        params = [params, params]

    rows, cols, ch = img.shape



    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])

    dst = cv2.warpAffine(img, M, (cols, rows))

    fmap = cv2.warpAffine(fmap, M, (cols, rows))

    return dst, fmap



def image_shear(img,fmap):

    params = np.random.randint(-20, 21)*0.01

    rows, cols, ch = img.shape

    factor = params*(-1.0)

    M = np.float32([[1, factor, 0], [0, 1, 0]])

    dst = cv2.warpAffine(img, M, (cols, rows))

    fmap = cv2.warpAffine(fmap, M, (cols, rows))

    return dst, fmap



def image_rotation(img,fmap):

    params = np.random.randint(-5, 6)

    rows, cols, ch = img.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)

    dst = cv2.warpAffine(img, M, (cols, rows))

    fmap = cv2.warpAffine(fmap, M, (cols, rows))

    return dst,fmap



def image_contrast(img,fmap):

    params = np.random.randint(7, 10)*0.1

    alpha = params

    dst = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha

    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

  

    return dst, fmap



def image_blur(img,fmap):

    params = params = np.random.randint(1, 21)

    blur = []

    if params == 1:

        blur = cv2.blur(img, (3, 3))

    if params == 2:

        blur = cv2.blur(img, (4, 4))

    if params == 3:

        blur = cv2.blur(img, (5, 5))

    if params == 4:

        blur = cv2.GaussianBlur(img, (3, 3), 0)

    if params == 5:

        blur = cv2.GaussianBlur(img, (5, 5), 0)

    if params == 6:

        blur = cv2.GaussianBlur(img, (7, 7), 0)

    if params == 7:

        blur = cv2.medianBlur(img, 3)

    if params == 8:

        blur = cv2.medianBlur(img, 5)

    if params == 9:

        blur = cv2.blur(img, (6, 6))

    if params == 10:

        blur = cv2.bilateralFilter(img, 9, 75, 75)

    if params > 10:

        blur = img

        

    return blur.reshape([blur.shape[0],blur.shape[1],1]), fmap
dst, fmap = vertical_flip(seed_image, seg_img)



plt.subplot(2, 1, 1)

plt.imshow(dst, "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray",vmin=0,vmax=1)
dst, fmap = horizontal_flip(seed_image, seg_img)



plt.subplot(2, 1, 1)

plt.imshow(dst)

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_translation(seed_image, seg_img)



plt.subplot(2, 1, 1)

plt.imshow(dst)

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_shear(seed_image, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst)

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_rotation(seed_image, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst)

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_contrast(seed_image, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst)

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
np.random.seed(2019)

np.random.shuffle(train_img)

train_num = int(len(train_img)*0.80)

train_idx = train_img[:train_num]

val_idx = train_img[train_num:]
len(train_idx)
len(val_idx)
img_width, img_height = 800, 128

num_train = len(train_idx)

num_val = len(val_idx)



pretrain_model = "xception"



batch_size = 8

print(num_train, num_val)

abs_path = "../input/severstal-steel-defect-detection/train_images/"
def get_segment_data(train, img_name, img_height, img_width):

    segment_4 = []

    for i in range(4):

        x = train[train["ImageId"] == img_name]

        x2 = x[x["ClassId"] == str(i+1)]

        x3 = x2["EncodedPixels"].values[0]



        if x3 ==0:

            x4 = "ok"



        else:

            x4 = x3.split()

            

        segment_4.append(x4)



    segment_4 = np.array(segment_4)

    

    #セグメンテーションの生成

    seg_img = np.zeros([img_height, img_width,5], dtype=np.uint8)



    for j in range(4):



        seg_np = np.zeros([img_height*img_width], dtype=np.uint8)



        if segment_4[j]=="ok":

            pass



        else:

            length=len(segment_4[j])//2

            for i in range(length):

                start = int(segment_4[j][2*i])

                length = int(segment_4[j][2*i+1])

                seg_np[start:start+length]=1



        seg_img[:,:,j+1] = seg_np.reshape([img_width,img_height]).T

        

    #seg_img[:,:,0] = np.ones([seed_image.shape[0], seed_image.shape[1]], dtype=np.uint8) - seg_img[:,:,1] - seg_img[:,:,2] - seg_img[:,:,3] - seg_img[:,:,4]

                

    return seg_img
def get_random_data(train_pd, img_index_1, abs_path, img_width, img_height, data_aug):

    image_file = abs_path + img_index_1

    

    seed_image = cv2.imread(image_file)

    fmap = get_segment_data(train_pd, img_index_1, img_height, img_width)

    

    seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

    fmap = cv2.resize(fmap, dsize=(img_width, img_height))

    

    if data_aug:

        

        r = np.random.rand()

        

        if r >= 0.5:

    

            seed_image, fmap = vertical_flip(seed_image, fmap)

            seed_image, fmap = horizontal_flip(seed_image, fmap)

            seed_image, fmap = image_shear(seed_image, fmap)

            seed_image, fmap = image_rotation(seed_image, fmap)

            seed_image, fmap = image_contrast(seed_image, fmap)

    

    seed_image = seed_image / 255

    

    fmap[:,:,0] = np.ones([img_height, img_width], dtype=np.float32) - fmap[:,:,1] - fmap[:,:,2] - fmap[:,:,3] - fmap[:,:,4]

    

    return seed_image, fmap
def data_generator(train_pd, img_index, batch_size, abs_path, img_width, img_height, data_aug):

    '''data generator for fit_generator'''

    n = len(img_index)

    i = 0

    while True:

        image_data = []

        fmap_data = []

        for b in range(batch_size):

            if i==0:

                np.random.shuffle(img_index)

            image, fmap = get_random_data(train_pd, img_index[i], abs_path, img_width, img_height, data_aug)

            image_data.append(image)

            fmap_data.append(fmap)

            i = (i+1) % n

        image_data = np.array(image_data)

        fmap_data = np.array(fmap_data)

        yield image_data, fmap_data



def data_generator_wrapper(train_pd, img_index, batch_size, abs_path, img_width, img_height, data_aug):

    n = len(img_index)

    if n==0 or batch_size<=0: return None

    return data_generator(train_pd, img_index, batch_size, abs_path, img_width, img_height, data_aug)
input_tensor = Input(shape=(img_height, img_width, 3))



if pretrain_model == "xception":



    basemodel = Xception(include_top=False, weights=None, input_tensor=input_tensor)



    basemodel.load_weights("../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

else:



    basemodel = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)



    basemodel.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")



basemodel.summary()
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
def RCU(data, filters, conv_sepa = True):

    

    if conv_sepa:

    

        x = BN()(data)

        x = Activation("relu")(x)

        x = SeparableConv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding="same")(x)

        x = BN()(x)

        x = Activation("relu")(x)

        x = SeparableConv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding="same")(x)



        x = shortcut_en(data, x)

        

    else:

    

        x = BN()(data)

        x = Activation("relu")(data)

        x = Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding="same")(x)

        x = BN()(x)

        x = Activation("relu")(x)

        x = Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding="same")(x)



        x = shortcut_en(data, x)

    

    return x
#re4 -> (8,50,2048)

#re3 -> (16,100,1024)

#re2 -> (32,200,512)

#re1 -> (64,400,256)



def Multi_Resolution_Fusion(re4, re3, re2, re1, conv_sepa = True):

    

    if conv_sepa:

    

        re3_shape = K.int_shape(re3)

        re4 = SeparableConv2D(filters=re3_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(re4)

        re4 = BN()(re4)

        re4 = UpSampling2D((2,2))(re4)

        re3_4 = Add()([re3, re4])



        re2_shape = K.int_shape(re2)

        re3_4 = SeparableConv2D(filters=re2_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(re3_4)

        re3_4 = BN()(re3_4)

        re3_4 = UpSampling2D((2,2))(re3_4)

        re2_3_4 = Add()([re2, re3_4])



        re1_shape = K.int_shape(re1)

        re2_3_4 = SeparableConv2D(filters=re1_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(re2_3_4)

        re2_3_4 = BN()(re2_3_4)

        re2_3_4 = UpSampling2D((2,2))(re2_3_4)

        re1_2_3_4 = Add()([re1, re2_3_4])

        

    else:

    

        re3_shape = K.int_shape(re3)

        re4 = Conv2D(filters=re3_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(re4)

        re4 = BN()(re4)

        re4 = UpSampling2D((2,2))(re4)

        re3_4 = Add()([re3, re4])



        re2_shape = K.int_shape(re2)

        re3_4 = Conv2D(filters=re2_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(re3_4)

        re3_4 = BN()(re3_4)

        re3_4 = UpSampling2D((2,2))(re3_4)

        re2_3_4 = Add()([re2, re3_4])



        re1_shape = K.int_shape(re1)

        re2_3_4 = Conv2D(filters=re1_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(re2_3_4)

        re2_3_4 = BN()(re2_3_4)

        re2_3_4 = UpSampling2D((2,2))(re2_3_4)

        re1_2_3_4 = Add()([re1, re2_3_4])

    

    return re1_2_3_4
def Chained_Residual_Pooling(data, conv_sepa = True):

    

    if conv_sepa:

    

        data_shape = K.int_shape(data)



        data = Activation("relu")(data)



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(data)

        x = SeparableConv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        data = Add()([data, x])



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)

        x = SeparableConv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        data = Add()([data, x])



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)

        x = SeparableConv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        data = Add()([data, x])



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)

        x = SeparableConv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        data = Add()([data, x])

        

    else:

    

        data_shape = K.int_shape(data)



        data = Activation("relu")(data)



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(data)

        x = Conv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        x = BN()(x)

        data = Add()([data, x])



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)

        x = Conv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        x = BN()(x)

        data = Add()([data, x])



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)

        x = Conv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        x = BN()(x)

        data = Add()([data, x])



        x = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)

        x = Conv2D(filters=data_shape[3],kernel_size=(3,3),strides=(1,1),padding="same")(x)

        x = BN()(x)

        data = Add()([data, x])

    

    return data
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
if pretrain_model == "xception":

    

    re1 = basemodel.get_layer("block3_sepconv2_bn").output

    re1 = ZeroPadding2D(padding=((1, 0), (1, 0)))(re1)



    re2 = basemodel.get_layer("block4_sepconv2_bn").output



    re3 = basemodel.get_layer("block13_sepconv2_bn").output



    re4 = basemodel.output

    

    re1 = RCU(re1, 256)

    re1 = RCU(re1, 256)



    re2 = RCU(re2, 256)

    re2 = RCU(re2, 256)



    re3 = RCU(re3, 256)

    re3 = RCU(re3, 256)



    re4 = RCU(re4, 512)

    re4 = RCU(re4, 512)



    re = Multi_Resolution_Fusion(re4, re3, re2, re1)

    re = Chained_Residual_Pooling(re)



    re = RCU(re, 256)



    x1 = Conv2DTranspose(filters=128,kernel_size=(1,1),strides=(2,2),padding="same",kernel_initializer='he_normal',

                              kernel_regularizer=l2(1.e-4))(re)



    x = UpSampling2D((2,2))(re)

    x=Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding="same")(x)

    x = BN()(x)

    x = Activation("relu")(x)



    x = Add()([x,x1])



    x2 = Conv2DTranspose(filters=64,kernel_size=(1,1),strides=(2,2),padding="same",kernel_initializer='he_normal',

                              kernel_regularizer=l2(1.e-4))(x)



    x = UpSampling2D((2,2))(x)

    x=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same")(x)

    x = BN()(x)

    x = Activation("relu")(x)



    x = Add()([x,x2])



    x=Conv2D(filters=5,kernel_size=(1,1),strides=(1,1),padding="same")(x)

    outputs = Activation('softmax')(x)

    

else:

    

    re1 = basemodel.get_layer("activation_10").output



    re2 = basemodel.get_layer("activation_22").output



    re3 = basemodel.get_layer("activation_40").output



    re4 = basemodel.output



    re1 = RCU(re1, 256, conv_sepa = False)

    re1 = RCU(re1, 256, conv_sepa = False)



    re2 = RCU(re2, 256, conv_sepa = False)

    re2 = RCU(re2, 256, conv_sepa = False)



    re3 = RCU(re3, 256, conv_sepa = False)

    re3 = RCU(re3, 256, conv_sepa = False)



    re4 = RCU(re4, 512, conv_sepa = False)

    re4 = RCU(re4, 512, conv_sepa = False)



    re = Multi_Resolution_Fusion(re4, re3, re2, re1, conv_sepa = False)

    re = Chained_Residual_Pooling(re, conv_sepa = False)

    

    re = RCU(re, 256)



    x = UpSampling2D((2,2))(re)

    x = Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding="same")(x)

    x = BN()(x)

    x = Activation("relu")(x)



    x = UpSampling2D((2,2))(x)

    x = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same")(x)

    x = BN()(x)

    x = Activation("relu")(x)

    

    x = Conv2D(filters=5,kernel_size=(1,1),strides=(1,1),padding="same")(x)

    outputs = Activation('softmax')(x)



# instantiate decoder model

model = Model(basemodel.input, outputs)

model.summary()



#model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model).create(prog='dot', format='svg'))
modelCheckpoint = ModelCheckpoint(filepath = 'best_weight.h5',

                                  monitor='val_acc',

                                  verbose=1,

                                  save_best_only=True,

                                  save_weights_only=True,

                                  mode='max',

                                  period=1)
start = time.time()



model.fit_generator(data_generator_wrapper(train,train_idx, batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(train,val_idx, batch_size, abs_path, img_width, img_height, False),

        validation_steps=max(1, num_val//batch_size),

        epochs=6,

        initial_epoch=0,

        callbacks=[modelCheckpoint])



elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
model.load_weights("best_weight.h5")
test_path = "../input/severstal-steel-defect-detection/test_images/"



test_list = os.listdir(test_path)



abs_name = test_path + test_list[0]

seed_image = cv2.imread(abs_name)

seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

seed_image = np.expand_dims(seed_image, axis=0)

seed_image = seed_image/255

pred = model.predict(seed_image)[0]
plt.figure(figsize=(15,15))

plt.imshow(seed_image[0,:,:,0], "gray")
pred = cv2.resize(pred, dsize=(1600, 256))
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15,15), sharey=True)

sns.heatmap(pred[:,:,0],vmin=0, vmax=1, ax=ax1)

sns.heatmap(pred[:,:,1],vmin=0, vmax=1, ax=ax2)

sns.heatmap(pred[:,:,2],vmin=0, vmax=1, ax=ax3)

sns.heatmap(pred[:,:,3],vmin=0, vmax=1, ax=ax4)

sns.heatmap(pred[:,:,4],vmin=0, vmax=1, ax=ax5)
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

    seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

    seed_image = np.expand_dims(seed_image, axis=0)

    seed_image = seed_image/255

    pred = model.predict(seed_image)[0]

    pred = cv2.resize(pred, dsize=(1600, 256))

    for i in range(4):

        

        pred_fi = pred[:,:,i+1].T.flatten()

        pred_fi = np.where(pred_fi > 0.25, 1, 0)

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

print(df)