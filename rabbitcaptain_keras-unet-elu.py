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



from keras.regularizers import l2



from keras.preprocessing.image import array_to_img, img_to_array, load_img



from sklearn.model_selection import train_test_split



from PIL import Image, ImageDraw, ImageFilter

print(os.listdir("../input"))
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

seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15,15))

plt.imshow(seed_image, "gray")
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
plt.figure(figsize=(15,15))

plt.imshow(seed_image, "gray")
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
np.random.seed(2019)

np.random.shuffle(train_img)

train_num = int(len(train_img)*0.80)

train_idx = train_img[:train_num]

val_idx = train_img[train_num:]
len(train_idx)
len(val_idx)
img_width, img_height = 1600, 256

num_train = len(train_idx)

num_val = len(val_idx)

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

    seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

    seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

    seed_image = np.expand_dims(seed_image, axis=-1)

    fmap = get_segment_data(train_pd, img_index_1, img_height, img_width)

    #fmap = cv2.resize(fmap, dsize=(img_width, img_height))

    

    if data_aug:

        

        r = np.random.rand()

        

        if r >= 0.5:

    

            seed_image, fmap = vertical_flip(seed_image, fmap)

            seed_image, fmap = horizontal_flip(seed_image, fmap)

            seed_image, fmap = image_shear(seed_image, fmap)

            seed_image, fmap = image_rotation(seed_image, fmap)

            seed_image, fmap = image_contrast(seed_image, fmap)

    

    seed_image = seed_image / 255

    

    fmap[:,:,0] = np.ones([img_height, img_width], dtype=np.uint8) - fmap[:,:,1] - fmap[:,:,2] - fmap[:,:,3] - fmap[:,:,4]

    

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
inputs = Input(shape=(img_height, img_width, 1))



c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (inputs)

c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (p1)

c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (p2)

c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (p3)

c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (p4)

c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (c5)

p5 = MaxPooling2D(pool_size=(2, 2)) (c5)



c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (p5)

c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (c55)



u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)

u6 = Concatenate()([u6, c5])

c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (u6)

c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (c6)



u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)

u71 = Concatenate()([u71, c4])

c71 = Conv2D(32, (3, 3), activation='elu', padding='same') (u71)

c61 = Conv2D(32, (3, 3), activation='elu', padding='same') (c71)



u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)

u7 = Concatenate()([u7, c3])

c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (u7)

c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (c7)



u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = Concatenate()([u8, c2])

c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (u8)

c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (c8)



u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = Concatenate()([u9, c1])

c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (u9)

c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (c9)



outputs = Conv2D(5, (1, 1), activation='sigmoid') (c9)



# instantiate decoder model

model = Model(inputs, outputs)

model.summary()



model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),

             loss="categorical_crossentropy", metrics=["accuracy"])
start = time.time()



model.fit_generator(data_generator_wrapper(train,train_idx, batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(train,val_idx, batch_size, abs_path, img_width, img_height, False),

        validation_steps=max(1, num_val//batch_size),

        epochs=10,

        initial_epoch=0)



elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
test_path = "../input/severstal-steel-defect-detection/test_images/"



test_list = os.listdir(test_path)



abs_name = test_path + test_list[3]

seed_image = cv2.imread(abs_name)

seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

seed_image = np.expand_dims(seed_image, axis=-1)

seed_image = np.expand_dims(seed_image, axis=0)

seed_image = seed_image/255

pred = model.predict(seed_image)
plt.figure(figsize=(15,15))

plt.imshow(seed_image[0,:,:,0], "gray")
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15,15), sharey=True)

sns.heatmap(pred[0,:,:,0],vmin=0, vmax=1, ax=ax1)

sns.heatmap(pred[0,:,:,1],vmin=0, vmax=1, ax=ax2)

sns.heatmap(pred[0,:,:,2],vmin=0, vmax=1, ax=ax3)

sns.heatmap(pred[0,:,:,3],vmin=0, vmax=1, ax=ax4)

sns.heatmap(pred[0,:,:,4],vmin=0, vmax=1, ax=ax5)
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

    seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

    seed_image = np.expand_dims(seed_image, axis=-1)

    seed_image = np.expand_dims(seed_image, axis=0)

    seed_image = seed_image/255

    pred = model.predict(seed_image)

    

    for i in range(4):

        

        pred_fi = pred[0,:,:,i+1].T.flatten()

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