import os #디렉토리를 조회하는 라이브러리

import sys

import random #랜덤변수 만드는

import warnings

import re





import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt #그림을 그려주는 라이브러리



from tqdm import tqdm, tqdm_notebook #돌고있는 내용을 보여주는



#케라스는 프레임워크

from keras.models import Model, load_model 

from keras.layers import Input

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras.optimizers import Adam

import tensorflow as tf



#싸이킷러닝(쉘로우 러닝에 적합)에서 파생된 함수가 밑에

from skimage.io import imread

from skimage.transform import resize
%matplotlib inline

# 아래는 쓰고 시작_그래야 잘 실행
TRAIN_PATH = '../input/train/'

TEST_PATH = '../input/test/'



seed = 42

random.seed = seed

np.random.seed = seed



#효율성을 위해 변수처리를 해서 진행

tot_num = 5635

IMG_HEIGHT = 128

IMG_WIDTH = 128 # 128 사이즈가 좋다/리사이즈 하기 위해



files = os.listdir(TRAIN_PATH) # 갯수 세기 위해서

masks_list = []

imgs_list = []



for f in files:

    if 'mask' in f:

        masks_list.append(f)

    else:

        imgs_list.append(f)
# MASKS 내림차순으로 정리하는 코드 

reg = re.compile("[0-9]+")



temp1 = list(map(lambda x: reg.match(x).group(), masks_list)) 

temp1 = list(map(int, temp1))



temp2 = list(map(lambda x: reg.match(x.split("_")[1]).group(), masks_list))

temp2 = list(map(int, temp2))



masks_list = [x for _,_,x in sorted(zip(temp1, temp2, masks_list))]

# IMGS 내림차순으로 정리하는 코드 

reg = re.compile("[0-9]+")



temp3 = list(map(lambda x: reg.match(x).group(), imgs_list)) 

temp3 = list(map(int, temp3))



temp4 = list(map(lambda x: reg.match(x.split("_")[1]).group(), imgs_list))

temp4 = list(map(int, temp4))



imgs_list = [x for _,_,x in sorted(zip(temp3, temp4, imgs_list))]
X_train = np.zeros((tot_num, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

# 이미지개수/높이/너비/ 데이터타입

Y_train = np.zeros((tot_num, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

#데이터타입을 불리언으로 해도 플롯으로 가져온다
for i, file in tqdm_notebook(enumerate(imgs_list), total=len(imgs_list)): #enumerate함수는 i_인덱스를 같이 뽑아가지고 온다

    img_path = file

    mask_path = img_path[:-4] + '_mask.tif' #밑에 imread에 불러오기위해 이름을 설정

   

    mask = imread(TRAIN_PATH + mask_path)

    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    Y_train[i] = mask



    img = imread(TRAIN_PATH + img_path)

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[i] = img
X_train = X_train/255 #RGB값이라 255/노말라이제이션 하는 과정

Y_train = Y_train/255
#이미지 확인 (plt)

plt.imshow(X_train[3], 'gray')
plt.imshow(Y_train[3], 'gray')
X_train_1 = []

Y_train_1 = []



def augmentation(imgs, masks): 

    for img, mask in zip(imgs, masks): #집은 이미지랑 마스크를 동시에 하나씩 뽑겠다/집을 안쓰면 뽑아낼 수 없다

        X_train_1.append(img)

        Y_train_1.append(img)

        img_lr = np.fliplr(img)

        mask_lr = np.fliplr(mask)

        img_up = np.flipud(img)

        mask_up = np.flipud(mask)

        img_lr_up = np.flipud(img_lr)

        mask_lr_up = np.flipud(mask_lr)

        img_up_lr = np.fliplr(img_up)

        mask_up_lr = np.fliplr(mask_up)

        X_train_1.append(img_lr)

        Y_train_1.append(mask_lr)

        X_train_1.append(img_up)

        Y_train_1.append(mask_up)

        X_train_1.append(img_lr_up)

        Y_train_1.append(mask_lr_up)

        X_train_1.append(img_up_lr)

        Y_train_1.append(mask_up_lr)
augmentation(X_train, Y_train)
len(X_train)
len(X_train_1) #Augmentation
X_train = np.array(X_train_1)

Y_train = np.array(Y_train_1)
X_train_ax = X_train[:,:,:,np.newaxis]

Y_train_ax = Y_train[:,:,:,np.newaxis] #채널값을 더하기 위해 축 하나더 생성
smooth = 1.



def dice_coef(y_true, y_pred): # 이함수는 실제 마스크랑 컴퓨터가 그린거랑 얼마나 겹치는지

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)





def dice_coef_loss(y_true, y_pred): #클수록 좋은 함수라 마이너스를 취한다.

    return -dice_coef(y_true, y_pred)
#바닐라 유넷

inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))



conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #32는 필터 갯수

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)



conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)



up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)

conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)

conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)



up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)

conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)

conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)



up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)

conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)

conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)



up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)

conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)

conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)



conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
model = Model(inputs=[inputs], outputs=[conv10]) #모델에 넣고

model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[dice_coef]) 
results = model.fit(X_train_ax, Y_train_ax, validation_split=0.1, batch_size=32, epochs=100) 

#10프로는 validation으로 사용
#  모델 학습 과정 표시하기

%matplotlib inline

import matplotlib.pyplot as plt



fig, loss_ax = plt.subplots()



acc_ax = loss_ax.twinx()



loss_ax.plot(results.history['loss'], 'y', label='train loss')

loss_ax.plot(results.history['val_loss'], 'r', label='val loss')



acc_ax.plot(results.history['dice_coef'], 'b', label='train dice coef')

acc_ax.plot(results.history['val_dice_coef'], 'g', label='val dice coef')



loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('loss')

acc_ax.set_ylabel('accuray')



loss_ax.legend(loc='upper left')

acc_ax.legend(loc='lower left')



plt.show()