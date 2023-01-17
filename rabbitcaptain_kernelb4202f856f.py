%matplotlib inline

import matplotlib.pyplot as plt

import random

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import tensorflow as tf



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

from keras.backend import tf as ktf



from keras.regularizers import l2



from keras.preprocessing.image import array_to_img, img_to_array, load_img



from sklearn.model_selection import train_test_split



from PIL import Image, ImageDraw, ImageFilter

print(os.listdir("../input"))
train = pd.read_csv('../input/understanding_cloud_organization/train.csv')
train['ImageId'] = train['Image_Label'].str[:11]

train['ClassId'] = train['Image_Label'].str[12:]

train = train[['ImageId','ClassId','EncodedPixels']]

train
train = train.fillna(0)
train
mask_count_df = train["ImageId"]

train_img = mask_count_df.drop_duplicates().reset_index()

train_img = train_img.drop("index", axis=1)

train_img
img_name = train["ImageId"][0]

img_name
abs_path = "../input/understanding_cloud_organization/train_images/"
seed_image = cv2.imread(abs_path+img_name)

seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15,15))

plt.imshow(seed_image, "gray")
seed_image.shape
df_exact = train[train["ImageId"] == img_name]

df_exact
filelist = os.listdir("../input/understanding_cloud_organization/train_images/")



n0 = 0

n1 = 0

n2 = 0

n3 = 0

n4 = 0



for i in filelist:

    x = train[train["ImageId"] == i]

    if len(x[x["EncodedPixels"] == 0]) == 0:

        n0 += 1

        

    elif len(x[x["EncodedPixels"] == 0]) == 1:

        n1 += 1

        

    elif len(x[x["EncodedPixels"] == 0]) == 2:

        n2 += 1

        

    elif len(x[x["EncodedPixels"] == 0]) == 3:

        n3 += 1

        

    elif len(x[x["EncodedPixels"] == 0]) == 4:

        n4 += 1

        

print(n0,n1,n2,n3,n4)
class_id = ["Fish","Flower","Gravel","Sugar"]



class_num = []



for i in range(4):



    x = train[train["ClassId"] == class_id[i]]

    class_num.append(len(x[x["EncodedPixels"] != 0]))

    

class_num = np.array(class_num)

print(class_num)
plt.bar(class_id, class_num)
df_exact2 = df_exact[df_exact["ClassId"] == "Fish"]

df_exact2
class_id = ["Fish","Flower","Gravel","Sugar"]



segment_4 = []

for i in range(4):

    x = train[train["ImageId"] == img_name]

    x2 = x[x["ClassId"] == class_id[i]]

    x3 = x2["EncodedPixels"].values[0]

    

    if x3 ==0:

        x4 = "ok"

        

    else:

        x4 = x3.split()

        

    segment_4.append(x4)



segment_4 = np.array(segment_4)
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

    #seg_img[:,:,j] = seg_np.reshape([seed_image.shape[0],seed_image.shape[1]])
plt.figure(figsize=(15,15))

plt.imshow(seed_image, "gray",vmin=0,vmax=255)
plt.figure(figsize=(15,15))

plt.imshow(seg_img[:,:,1],"gray",vmin=0,vmax=1)
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

    params = np.random.randint(5, 15)*0.1

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



plt.figure(figsize=(15,15))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray",vmin=0,vmax=1)
dst, fmap = horizontal_flip(seed_image2, seg_img)



plt.figure(figsize=(15,15))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_translation(seed_image2, seg_img)



plt.figure(figsize=(15,15))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_shear(seed_image2, seg_img)

plt.figure(figsize=(15,15))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_rotation(seed_image2, seg_img)

plt.figure(figsize=(15,15))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_contrast(seed_image2, seg_img)

plt.figure(figsize=(15,15))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
dst, fmap = image_blur(seed_image2, seg_img)

plt.figure(figsize=(15,15))

plt.subplot(2, 1, 1)

plt.imshow(dst[:,:,0], "gray")

plt.subplot(2, 1, 2)

plt.imshow(fmap[:,:,0], "gray")
train_idx, val_idx = train_test_split(

    train_img.index, random_state=2019, test_size=0.15

)
img_width, img_height = 525, 350 

num_train = len(train_idx)

num_val = len(val_idx)

batch_size = 4

print(num_train, num_val)

abs_path = "../input/understanding_cloud_organization/train_images/"
def get_segment_data(train, img_name, class_id = ["Fish","Flower","Gravel","Sugar"]):

    segment_4 = []

    for i in range(4):

        x = train[train["ImageId"] == img_name]

        x2 = x[x["ClassId"] == class_id[i]]

        x3 = x2["EncodedPixels"].values[0]



        if x3 ==0:

            x4 = "ok"



        else:

            x4 = x3.split()



        segment_4.append(x4)



    segment_4 = np.array(segment_4)

    

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

                

    return seg_img
def get_random_data(train_pd, img_index_1, abs_path, img_width, img_height, data_aug):

    image_name = train_img["ImageId"][img_index_1]

    image_file = abs_path + image_name

    fmap = get_segment_data(train_pd, image_name)

    

    seed_image = cv2.imread(image_file)

    seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2GRAY)

    seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

    seed_image = np.expand_dims(seed_image, axis=-1)

    fmap = cv2.resize(fmap, dsize=(img_width, img_height))

    

    if data_aug:

        

        r = np.random.rand()

        

        if r >= 0.5:

    

            seed_image, fmap = vertical_flip(seed_image, fmap)

            seed_image, fmap = horizontal_flip(seed_image, fmap)

            seed_image, fmap = image_shear(seed_image, fmap)

            seed_image, fmap = image_rotation(seed_image, fmap)

            seed_image, fmap = image_contrast(seed_image, fmap)

            seed_image, fmap = image_blur(seed_image, fmap)

    

    seed_image = seed_image / 255

    

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

                img_index = img_index.take(np.random.permutation(len(img_index)))

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
def resnet_en(data, filters, kernel_size, dilation_rate,option=False):

    if option:

        x=BN()(data)

        x = Activation("relu")(x)

        x=Conv2D(filters=filters,kernel_size=kernel_size,dilation_rate=(1,1),strides=(2,2),padding="same")(x)



        x=BN()(x)

        x = Activation("relu")(x)

        x=Conv2D(filters=filters,kernel_size=kernel_size,dilation_rate=dilation_rate,strides=(1,1),padding="same")(x)

        

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



x = Conv2D(filters=16,kernel_size=(7,7), strides=1,padding='same')(inputs)

x = BN()(x)

x = Activation("relu")(x)



fx = resnet_en(x, 16, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 16, (3,3), (1,1), True)

x = shortcut_en(x, fx)



fx = resnet_en(x, 32, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 32, (3,3), (1,1), True)

x = shortcut_en(x, fx)



fx = resnet_en(x, 64, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 64, (3,3), (1,1))

x = shortcut_en(x, fx)



fx = resnet_en(x, 64, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 64, (3,3), (1,1), True)

x = shortcut_en(x, fx)



fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)



fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)

fx = resnet_en(x, 128, (3,3), (1,1))

x = shortcut_en(x, fx)



fx = resnet_en(x, 256, (3,3), (2,2))

x = shortcut_en(inputs, fx)

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



x1 = MaxPooling2D((1,1),padding="same")(x)

x1=Conv2DTranspose(filters=512//4,kernel_size=(1,1),strides=(1,1),padding="same")(x1)

x1 = BN()(x1)

x1 = Activation("relu")(x1)



x2 = MaxPooling2D((2,2),padding="same")(x)

x2=Conv2DTranspose(filters=512//4,kernel_size=(1,1),strides=(2,2),padding="same")(x2)

x2 = BN()(x2)

x2 = Activation("relu")(x2)



x3 = MaxPooling2D((4,6),padding="same")(x)

x3=Conv2DTranspose(filters=512//4,kernel_size=(1,1),strides=(4,6),padding="same")(x3)

x3 = BN()(x3)

x3 = Activation("relu")(x3)



x4 = MaxPooling2D((11,11),padding="same")(x)

x4=Conv2DTranspose(filters=512//4,kernel_size=(1,1),strides=(11,11),padding="same")(x4)

x4 = BN()(x4)

x4 = Activation("relu")(x4)



x = Concatenate()([x,x1,x2,x3,x4])



x = UpSampling2D((2,2))(x)

x=Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = UpSampling2D((2,2))(x)

x=Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = UpSampling2D((2,2))(x)

x=Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding="same")(x)

x = BN()(x)

x = Activation("relu")(x)



x = Lambda(lambda x: ktf.image.resize_images(x, [img_height, img_width],

                                          align_corners=True), output_shape=(350,525,128))(x)



x=Conv2D(filters=4,kernel_size=(1,1),strides=(1,1),padding="same")(x)

outputs = Activation('softmax')(x)



# instantiate decoder model

model = Model(inputs, outputs)

model.summary()



model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),

             loss="categorical_crossentropy", metrics=["accuracy"])
model.fit_generator(data_generator_wrapper(train,train_idx, batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(train,train_idx, batch_size, abs_path, img_width, img_height, False),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0)
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
test_path = "../input/understanding_cloud_organization/test_images/"



test_list = os.listdir(test_path)



class_id = ["Fish","Flower","Gravel","Sugar"]



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

        

        pred_fi = pred[0,:,:,i].T.flatten()

        pred_fi = np.where(pred_fi > 0.1, 1, 0)

        pred_fi_id = np.where(pred_fi == 1)

        pred_fi_id = make_testdata(pred_fi_id[0])

        x = np.array([fn + "_" + class_id[i],pred_fi_id])

        data.append(x)

    

data = np.array(data)
columns = ['Image_Label', 'EncodedPixels']

d = pd.DataFrame(data=data, columns=columns, dtype='str')

d.to_csv("submission.csv",index=False)

df = pd.read_csv("submission.csv")

print(df)