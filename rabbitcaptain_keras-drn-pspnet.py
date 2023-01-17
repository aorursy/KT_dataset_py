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
train = pd.read_csv('../input/train.csv')
train['ImageId'] = train['ImageId_ClassId'].str[:-2]

train['ClassId'] = train['ImageId_ClassId'].str[-1:]

train = train[['ImageId','ClassId','EncodedPixels']]

train
train = train.fillna(0)
train
start = time.time()



filelist = os.listdir("../input/train_images/")



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
abs_path = "../input/train_images/"
seed_image = cv2.imread(abs_path+img_name)

seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15,15))

plt.imshow(seed_image, "gray")
seed_image_resize = cv2.resize(seed_image, dsize=(1600, 256))

plt.figure(figsize=(15,15))

plt.imshow(seed_image_resize, "gray")
seed_image.shape
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

seg_img = np.ones([seed_image.shape[0], seed_image.shape[1],5], dtype=np.uint8)



for j in range(4):

    

    seg_np = np.ones([seed_image.shape[0]*seed_image.shape[1]], dtype=np.uint8)

    

    if segment_4[j]=="ok":

        pass

    

    else:

        for i in range(len(segment_4[j])//2):

            start = int(segment_4[j][2*i])

            length = int(segment_4[j][2*i+1])

            seg_np[start:start+length]=0



    seg_img[:,:,j+1] = seg_np.reshape([seed_image.shape[1],seed_image.shape[0]]).T
seg_img[:,:,0] = seg_img[:,:,0]*4 - seg_img[:,:,1] - seg_img[:,:,2] - seg_img[:,:,3] - seg_img[:,:,4]
seed_image = cv2.resize(seed_image, dsize=(800, 128))

seg_img = cv2.resize(seg_img, dsize=(800, 128))
plt.figure(figsize=(15,15))

plt.imshow(seed_image, "gray",vmin=0,vmax=255)
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

    return np.expand_dims(dst, axis=-1), fmap



def image_shear(img,fmap):

    params = np.random.randint(-20, 21)*0.01

    rows, cols, ch = img.shape

    factor = params*(-1.0)

    M = np.float32([[1, factor, 0], [0, 1, 0]])

    dst = cv2.warpAffine(img, M, (cols, rows))

    fmap = cv2.warpAffine(fmap, M, (cols, rows))

    return np.expand_dims(dst, axis=-1), fmap



def image_rotation(img,fmap):

    params = np.random.randint(-5, 6)

    rows, cols, ch = img.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)

    dst = cv2.warpAffine(img, M, (cols, rows))

    fmap = cv2.warpAffine(fmap, M, (cols, rows))

    return np.expand_dims(dst, axis=-1),fmap



def image_contrast(img,fmap):

    params = np.random.randint(7, 10)*0.1

    alpha = params

    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha

    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

  

    return np.expand_dims(new_img, axis=-1), fmap



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



def image_bitwise_not(image,fmap, rate=0.5):

    if np.random.rand() < rate:

        image = cv2.bitwise_not(image)

        image = np.expand_dims(image, axis=-1)

    return image, fmap
seed_image2 = np.expand_dims(seed_image, axis=-1)
dst, fmap = vertical_flip(seed_image2, seg_img)



plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray",vmin=0,vmax=1)
dst, fmap = horizontal_flip(seed_image2, seg_img)



plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_translation(seed_image2, seg_img)



plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_shear(seed_image2, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_rotation(seed_image2, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_contrast(seed_image2, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_blur(seed_image2, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_bitwise_not(seed_image2, seg_img)

plt.figure(figsize=(15,5))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

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

batch_size = 8

print(num_train, num_val)

abs_path = "../input/train_images/"
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

    seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

    fmap = get_segment_data(train_pd, img_index_1, img_height*2, img_width*2)

    seed_image = np.expand_dims(seed_image, axis=-1)

    

    if data_aug:

        

        r = np.random.rand()

        

        if r >= 0.5:

    

            seed_image, fmap = vertical_flip(seed_image, fmap)

            seed_image, fmap = horizontal_flip(seed_image, fmap)

            seed_image, fmap = image_shear(seed_image, fmap)

            seed_image, fmap = image_rotation(seed_image, fmap)

            seed_image, fmap = image_contrast(seed_image, fmap)

            

    seed_image_resize = cv2.resize(seed_image, dsize=(img_width, img_height))

    seed_image_resize = np.expand_dims(seed_image_resize, axis=-1)

    

    seed_image = seed_image / 255

    seed_image_resize = seed_image_resize / 255

    

    fmap[:,:,0] = np.ones([img_height*2, img_width*2], dtype=np.float32) - fmap[:,:,1] - fmap[:,:,2] - fmap[:,:,3] - fmap[:,:,4]

    

    return seed_image_resize, seed_image, fmap
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

        epochs=5,

        initial_epoch=0,

        callbacks=[modelCheckpoint])



elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
model.load_weights("best_weight.h5")
test_path = "../input/test_images/"



test_list = os.listdir(test_path)



abs_name = test_path + test_list[3]

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
plt.figure(figsize=(15,15))

plt.imshow(seed_image[0,:,:,0], "gray")
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



test_path = "../input/test_images/"



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

print(df)