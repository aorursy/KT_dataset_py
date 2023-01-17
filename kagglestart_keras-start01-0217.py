import os

print(os.listdir("../input"))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2, random

from matplotlib import pyplot as plt

%matplotlib inline
# numpy random 발생기 지정.

np.random.seed(1339)
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'
img = cv2.imread('../input/train/dog.3.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

print(img.shape)

print(len(os.listdir(TRAIN_DIR))) # 25001장 

print(len(os.listdir(TEST_DIR)))  # 12501장
print(img.shape)

img
print(os.listdir(TRAIN_DIR)[:5])

print(os.listdir(TEST_DIR)[0:5])
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]



print(train_images[0:3], train_dogs[0:3], train_cats[0:3])

print(test_images[0:3])
print(train_images[24990:]) 

#random.shuffle(train_images)  # 이미지를 섞기

#print(train_images[0:10])
print(cv2.IMREAD_COLOR)

print(cv2.IMREAD_GRAYSCALE)

print(cv2.IMREAD_UNCHANGED)
print(cv2.COLOR_BGR2RGB)

print(cv2.INTER_CUBIC)
ROWS = 100      # 이미지 세로 

COLS = 100      # 이미지 가로

CHANNELS = 3    # 이미지 채널 수



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) # (1) 컬러 이미지로 읽기

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # (2) BGR를 RGB로 변경하기

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC) # (3) 이미지 리사이즈 하기
## 이미지 파일 리스트를 받아, 이미지를 리사이즈하고,

## 이미지 정보를 0-1 사이의 값으로 바꾸는 것 수행하기

## 이를 수행한, 반환

def prep_data(images):

    count = len(images)

    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.float32)



    for i, image_file in enumerate(images):

        if i>24490:

            print(i, image_file)

        image = read_image(image_file)

        image = image / 255

        data[i] = image

        if i%500 == 0: print('Processed {} of {}'.format(i, count))

    

    return data
%%time



#train = prep_data(train_images)

#test = prep_data(test_images)
#print("Train shape: {}".format(train.shape))

#print("Test shape: {}".format(test.shape))
## 문제가 있다면 답이 있어야 하므로, 이미지 이름에서 dog, cat를 분류하자.

#labels = []

#for i in train_images:

#    if 'dog' in i:

#        labels.append(1)

#    else:

#        labels.append(0)

#labels = np.asarray(labels).astype('float32')