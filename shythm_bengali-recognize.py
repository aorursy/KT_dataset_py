import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import gc

import tensorflow as tf

import time

import keras

import cv2

import scipy.special

from keras.optimizers import SGD

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Activation, Add, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization, Lambda

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.regularizers import l2

from keras.utils import plot_model

from tqdm.auto import tqdm



from PIL import Image



# set max display columns and rows count

pd.set_option('display.max_columns', 10000)

pd.set_option('display.max_rows', 10000)
import os

__print__ = print

def print_log(string):

    os.system(f'echo \"{string}\"')

    __print__(string)
# Install EfficientNet

!pip install '/kaggle/input/kerasefficientnetb3/efficientnet-1.0.0-py3-none-any.whl'

import efficientnet.keras as efn
# class_map 불러오기(grapheme_root, vowel_diacritic, consonant_diacritic 종류와 딕셔너리가 들어있음)

class_map = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')



# class_map에 있는 grapheme_root, vowel_diacritic, consonant_diacritic 정보를 분리하여 보관

grapheme_root = class_map[class_map['component_type'] == 'grapheme_root']['component'].values

GRAPHEME_ROOT_NUM = grapheme_root.shape[0]

vowel_diacritic = class_map[class_map['component_type'] == 'vowel_diacritic']['component'].values

VOWEL_DIACRITIC_NUM = vowel_diacritic.shape[0]

consonant_diacritic = class_map[class_map['component_type'] == 'consonant_diacritic']['component'].values

CONSONANT_DIACRITIC_NUM = consonant_diacritic.shape[0]
# train.csv 파일 경로

TRAIN_META_PATH = '/kaggle/input/bengaliai-cv19/train.csv'



# train_image_data.parquet 파일 경로

TRAIN_IMG_PATH = ['/kaggle/input/bengaliai-cv19/train_image_data_0.parquet',

           '/kaggle/input/bengaliai-cv19/train_image_data_1.parquet',

           '/kaggle/input/bengaliai-cv19/train_image_data_2.parquet',

           '/kaggle/input/bengaliai-cv19/train_image_data_3.parquet']



# 이미지 크기 정보(image height, image width)

RAW_IMG_ROWS, RAW_IMG_COLUMNS = 137, 236
# 학습 레이블 가져오는 함수

# split_y 인자를 통해 train_y를 3개의 train_y_grapheme_root, train_y_vowel_diacritic, train_y_consonant_diacritic로 나눌 수 있다

def get_train_y(train_meta_path, data_range, split_y=True):

    train_meta_data = pd.read_csv(train_meta_path)

    

    # pandas 내장 함수를 이용해서 one hot encoding 적용

    train_y_grapheme_root       = pd.get_dummies(train_meta_data['grapheme_root']).to_numpy(dtype='float32')

    train_y_vowel_diacritic     = pd.get_dummies(train_meta_data['vowel_diacritic']).to_numpy(dtype='float32')

    train_y_consonant_diacritic = pd.get_dummies(train_meta_data['consonant_diacritic']).to_numpy(dtype='float32')

    

    # multiclassification: 마지막 레이어에 들어갈 레이블을 서로 분리할 지, 합칠 지를 적용

    train_y = [train_y_grapheme_root[data_range], train_y_vowel_diacritic[data_range], train_y_consonant_diacritic[data_range]]

    if (split_y is False):

        train_y = np.concatenate(train_y, axis=1)

        

    return train_y



# train_image_data_x.parquet을 불러와서 numpy 배열로 되어 있는 이미지와, 시작 Id, 그리고 들어 있는 이미지의 개수를 반환한다.

def get_img_data(img_path):

    data = pd.read_parquet(img_path)

    

    # train_image_data의 인덱스 정보와 Image_Id 정보가 일치하지 않을 수 있음(train_image_data_1.parquet의 index가 0일 때에는 Image_Id가 50210이다.)

    # 따라서 meta data와 상응한 정보를 가져오게 하기 위해서 train_image_data의 첫 id 값과 요소의 개수를 반환하도록 한다.

    start_id    = int(data.iloc[0, 0].split('_')[1])

    element_num = data.shape[0]

    

    # pandas로 데이터를 불러와 numpy 배열로 바꾸는 작업이 필요하다.

    # pandas로 데이터를 조작할 때에는 속도가 너무 느리다(특히 iloc). 따라서 numpy 배열로 바꾸어 반환해서 속도 향상을 꾀한다.

    img_data = data[data.columns[1:]]

    img_data = img_data.to_numpy(dtype='uint8')

    img_data = img_data.reshape(-1, RAW_IMG_ROWS, RAW_IMG_COLUMNS)

    

    del data

    gc.collect()

    

    return img_data, start_id, element_num
# 데이터 전처리 함수 Version 1

def load_train_data_v1(train_img_path, split_y=True):

    print('학습 이미지 불러오는 중 ...')

    print('  경로:', train_img_path)

    img_data, start_id, element_num = get_img_data(train_img_path)

    print('  총 ', element_num, '개의 이미지')

    print('학습 이미지 불러오기 완료 !')

    

    # create train_y

    print('학습 레이블 생성 중 ...')

    train_y = get_train_y(TRAIN_META_PATH, range(start_id, start_id + element_num), split_y=split_y)

    print('학습 레이블 생성 완료 !')



    # create train_x

    print('학습 이미지 생성중 ...')

    # train_x 정보 생성 및 이미지 크기 조절

    img_rows = RAW_IMG_ROWS // 2

    img_columns = RAW_IMG_COLUMNS // 2

    train_x = np.empty(dtype='uint8', shape=(element_num, img_rows, img_columns))

    

    for i in range(train_x.shape[0]): # resize image

        train_x[i] = cv2.resize(img_data[i], dsize=(img_columns, img_rows)) # width * height

    print('  학습 이미지 크기:', '(' + str(img_rows) + ', ' + str(img_columns) + ')')

    

    # 메모리 관리

    del img_data # img_data를 더 이상 안 쓰므로 일단 메모리에서 삭제

    gc.collect()



    # keras CNN 모델에 넣기 위한 차원 및 데이터타입 정리

    train_x = train_x.reshape((-1, img_rows, img_columns, 1))

    train_x = train_x.astype('float32')

    train_x = train_x / 255.0

    print('학습 이미지 생성 완료 !')



    return train_x, train_y
# 데이터 전처리 함수 Version 2

def load_train_data_v2(train_img_path, output_img_size, split_y=True):

    print('학습 이미지 불러오는 중 ...')

    print('  경로:', train_img_path)

    img_data, start_id, element_num = get_img_data(train_img_path)

    print('  총 ', element_num, '개의 이미지')

    print('학습 이미지 불러오기 완료 !')

    

    # create train_y

    print('학습 레이블 생성 중 ...')

    train_y = get_train_y(TRAIN_META_PATH, range(start_id, start_id + element_num), split_y=split_y)

    print('학습 레이블 생성 완료 !')

    

    # create train_x

    print('학습 이미지 생성중 ...')

    img_rows    = output_img_size[1]

    img_columns = output_img_size[0]

    train_x = np.empty(dtype='uint8', shape=(element_num, img_rows, img_columns))

    

    for i in tqdm(range(train_x.shape[0])):

        img = img_data[i]

        # get ROI(region of interest)

        x1, y1, x2, y2 = get_img_roi(img)

        # crop image

        img = img[y1:y2, x1:x2]

        # resize image

        img = cv2.resize(img, dsize=(img_columns, img_rows)) # width * height

        train_x[i] = img

        

    print('  학습 이미지 크기:', '(' + str(img_rows) + ', ' + str(img_columns) + ')')

    

    # keras CNN 모델에 넣기 위한 차원 및 데이터타입 정리

    train_x = train_x.reshape((-1, img_rows, img_columns, 1))

    train_x = train_x.astype('float32')

    train_x = train_x / 255.0

    print('학습 이미지 생성 완료!')

    

    return train_x, train_y



# 이미지의 관심 영역을 좌표(x1, y1, x2, y2)로 반환하는 함수

def get_img_roi(img):

    _, img_thres = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        

    img_thres_sum_x = img_thres.sum(axis=0)

    for x1 in range(0, RAW_IMG_COLUMNS, 1):

        if (img_thres_sum_x[x1] > 0):

            break



    for x2 in range(RAW_IMG_COLUMNS-1, -1, -1):

        if (img_thres_sum_x[x2] > 0):

            break



    img_thres_sum_y = img_thres.sum(axis=1)

    for y1 in range(0, RAW_IMG_ROWS, 1):

        if (img_thres_sum_y[y1] > 0):

            break



    for y2 in range(RAW_IMG_ROWS-1, -1, -1):

        if (img_thres_sum_y[y2] > 0):

            break

            

    return x1, y1, x2, y2
IMG_ROWS    = 64

IMG_COLUMNS = 64



train_x, train_y = load_train_data_v2(TRAIN_IMG_PATH[0], output_img_size=(IMG_ROWS, IMG_COLUMNS), split_y=True)
# 특정 벵골어 손글씨 데이터 시각화(train set에서만 사용)

def visualize_grapheme(grapheme, train_img_path):

    meta_data = pd.read_csv(TRAIN_META_PATH)

    img_data = pd.read_parquet(train_img_path)

    

    grapheme_id = meta_data[meta_data['grapheme'] == grapheme]['image_id'].values # train_data에서 표시할 벵골어에 해당하는 image_id 선택

    grapheme_image = img_data[img_data['image_id'].isin(grapheme_id)] # train_image_data에서 표시할 벵골어에 해당하는 이미지 선택

    size = len(grapheme_image.index) # 이미지 개수



    # matplotlib figure 설정

    columns = 8

    rows = size / columns + 1

    fig = plt.figure(figsize=(30, rows * 3))



    # figure 그리기(imshow 이용, imshow를 사용할 때 데이터 타입이 uint8이어야 한다.)

    for i in range(size):

        image_index = str(grapheme_image.iloc[i, 0]).split('_')[1]

        image = grapheme_image.iloc[i].values[1:].astype('uint8').reshape(RAW_IMG_ROWS, RAW_IMG_COLUMNS)



        ax = fig.add_subplot(rows, columns, i + 1)

        ax.imshow(image, cmap='gray')

        ax.set_xlabel(grapheme_id[i])



    plt.show()

    del(meta_data)

    del(img_data)

    gc.collect()

            

# 벵골어 손글씨 데이터 시각화

def visualize_bengali(img_path, count, bias = 0):

    img_data = pd.read_parquet(img_path)

    

    # 뱅골어 손글씨 데이터 시각화 (matplotlib 한 화면에 여러개 그래프 그리기를 통해)

    columns = 8

    rows = int(count / columns) + 1

    fig = plt.figure(figsize=(30, int(rows * 3)))



    for i in range(0, count):

        image_index = str(img_data.iloc[bias + i, 0]).split('_')[1]

        # imshow를 하기 위해서는 nparray의 type이 uint8이어야 한다.

        image = img_data.iloc[bias + i].values[1:].astype('uint8').reshape(RAW_IMG_ROWS, RAW_IMG_COLUMNS)



        ax = fig.add_subplot(rows, columns, i + 1)

        ax.imshow(image, cmap='gray')

        ax.set_xlabel(image_index)



    plt.show()

    del(img_data)

    gc.collect()

    

# grapheme를 입력하면 시각화해주고, grapheme의 정보를 출력해준다.

def show_grapheme_info(x, y, num, size):

    plt.imshow(x[num].reshape(size), cmap='gray')

    if (y is not list):

        y = np.concatenate(y, axis=1)

    

    print(class_map[y[num] == 1])
show_grapheme_info(x=train_x, y=train_y, num=100, size=(IMG_ROWS, IMG_COLUMNS))

# visualize_grapheme('লা', TRAIN_IMG_PATH[0])

# visualize_bengali(TRAIN_IMG_PATH[2], count=24, bias=11160)
INPUT_SHAPE = (IMG_ROWS, IMG_COLUMNS, 1)



# activation functions

relu = lambda x: keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)

swish = lambda x: x * keras.activations.sigmoid(x)
# resnet convolution unit

def resnet_conv2d_unit(filters, kernel_size, strides):

    return Conv2D(filters=filters,

                  kernel_size=kernel_size,

                  strides=strides,

                  padding='same',

                  kernel_initializer='he_normal',

                  kernel_regularizer=l2(1e-4))
def get_resnet_v1_conv1_layer(x, activation):

    x = resnet_conv2d_unit(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

    x = BatchNormalization()(x)

    x = Activation(activation)(x)

    return x



def get_resnet_v1_conv2_layer(x, activation, count):

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    

    for i in range(count):

        shortcut = x

        

        x = resnet_conv2d_unit(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)

        

        x = resnet_conv2d_unit(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Add()([x, shortcut])

        x = Activation(activation)(x)

        

    return x



def get_resnet_v1_conv3_to_5_layer(x, activation, filters, count):

    for i in range(count):

        shortcut = x

        

        if i == 0:

            x = resnet_conv2d_unit(filters=filters, kernel_size=(3, 3), strides=(2, 2))(x)

        else:

            x = resnet_conv2d_unit(filters=filters, kernel_size=(3, 3), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)

        

        x = resnet_conv2d_unit(filters=filters, kernel_size=(3, 3), strides=(1, 1))(x)

        x = Dropout(rate=0.2)(x)

        x = BatchNormalization()(x)

        if i == 0:

            shortcut = resnet_conv2d_unit(filters=filters, kernel_size=(1, 1), strides=(2, 2))(shortcut)

            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])

        x = Activation(activation)(x)

        

    return x
# ResNet v1 구현

def get_resnet_v1_model(layer_num=(2, 2, 2, 2)):

    input_tensor = Input(shape=INPUT_SHAPE, dtype='float32')



    layer = get_resnet_v1_conv1_layer(input_tensor, activation=swish)

    layer = get_resnet_v1_conv2_layer(layer, activation=swish, count=layer_num[0]) # filter_size: 64

    layer = get_resnet_v1_conv3_to_5_layer(layer, activation=swish, filters=128, count=layer_num[1])

    layer = get_resnet_v1_conv3_to_5_layer(layer, activation=swish, filters=256, count=layer_num[2])

    layer = get_resnet_v1_conv3_to_5_layer(layer, activation=swish, filters=512, count=layer_num[3])



    layer = Flatten()(layer)

    layer = BatchNormalization()(layer)

    layer = Activation(activation=swish)(layer)

    

    layer = Dense(1024, activation=swish, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(layer)

    layer = Dropout(0.2)(layer)

    layer = BatchNormalization()(layer)

    layer = Activation(activation=swish)(layer)

    

    # !! 시행 착오 !!

    # * 다른 곳에서는 초기값을 설정할 때 he 방법을 사용했는데, 마지막 output network의 가중치를 초기화 할 때

    #   kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)를 사용해주지 않았다.

    #   가중치를 검증된 방법을 초기화 하니까 훨씬 더 수렴이 잘 되었다.



    # grapheme_root

    output_grapheme_root       = Dense(GRAPHEME_ROOT_NUM,

                                       name='grapheme_root',

                                       activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(layer)

    # vowel_diacritic

    output_vowel_diacritic     = Dense(VOWEL_DIACRITIC_NUM,

                                       name='vowel_diacritic',

                                       activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(layer)

    # consonant_diacritic

    output_consonant_diacritic = Dense(CONSONANT_DIACRITIC_NUM,

                                       name='consonant_diacritic',

                                       activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(layer)



    return Model(inputs=input_tensor,

                 outputs=[output_grapheme_root, output_vowel_diacritic, output_consonant_diacritic])
# ResNet v2 first layer

# (conv1: 7x7_64_stride2)

def get_resnet_v2_conv1_layer(x, activation):

    layer = resnet_conv2d_unit(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

    layer = BatchNormalization()(layer)

    layer = Activation(activation)(layer)

    return layer



# ResNet v2 second layer

# (conv2: 3x3maxpooling_stride2 -> 1x1_64 -> 3x3_64 -> 1x1_256)

def get_resnet_v2_conv2_layer(x, activation, count):

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    

    for i in range(count):

        shortcut = x # for skip connection(스킵 연결)

        

        x = resnet_conv2d_unit(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=256, kernel_size=(1, 1), strides=(1, 1))(x)

        x = BatchNormalization()(x)



        # Skip Connection 할 때에는 출력 텐서의 차원과 shortcut 텐서의 차원이 서로 틀리므로 같게 만들어준다.

        if (i == 0):

            shortcut = resnet_conv2d_unit(filters=256, kernel_size=(1, 1), strides=(1, 1))(shortcut) 

        

        x = Add()([x, shortcut]) # Skip Connection

        x = Activation(activation)(x)



    return x



# ResNet v2 third layer

# (conv3: 1x1_128(stride2 at i==0) -> 3x3_128 -> 1x1_512

def get_resnet_v2_conv3_layer(x, activation, count):

    for i in range(count):

        shortcut = x # for skip connection(스킵 연결)

        

        # output size 축소(가로 세로 절반으로)

        if (i == 0):

            x = resnet_conv2d_unit(filters=128, kernel_size=(1, 1), strides=(2, 2))(x)

        else:

            x = resnet_conv2d_unit(filters=128, kernel_size=(1, 1), strides=(1, 1))(x)



        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=512, kernel_size=(1, 1), strides=(1, 1))(x)

        x = BatchNormalization()(x)



        # 처음 Skip Connection 할 때에는 출력 텐서의 차원과 shortcut 텐서의 차원이 서로 틀리므로 같게 만들어준다.

        # output size도 다르므로 stride를 2로 설정해서 output size의 크기를 1/4로 줄인다(가로와 세로가 둘 다 반씩 줄었으므로)

        if (i == 0):

            shortcut = resnet_conv2d_unit(filters=512, kernel_size=(1, 1), strides=(2, 2))(shortcut) 



        x = Add()([x, shortcut]) # Skip Connection

        x = Activation(activation)(x)



    return x



# ResNet v2 fourth layer

# (conv4: 1x1_256(stride2 at i==0) -> 3x3_256 -> 1x1_1024)

def get_resnet_v2_conv4_layer(x, activation, count):

    for i in range(count):

        shortcut = x # for skip connection(스킵 연결)

        

        # output size 축소(가로 세로 절반으로)

        if (i == 0):

            x = resnet_conv2d_unit(filters=256, kernel_size=(1, 1), strides=(2, 2))(x)

        else:

            x = resnet_conv2d_unit(filters=256, kernel_size=(1, 1), strides=(1, 1))(x)



        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=256, kernel_size=(3, 3), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=1024, kernel_size=(1, 1), strides=(1, 1))(x)

        x = BatchNormalization()(x)



        # 처음 Skip Connection 할 때에는 출력 텐서의 차원과 shortcut 텐서의 차원이 서로 틀리므로 같게 만들어준다.

        # output size도 다르므로 stride를 2로 설정해서 output size의 크기를 1/4로 줄인다(가로와 세로가 둘 다 반씩 줄었으므로)

        if (i == 0):

            shortcut = resnet_conv2d_unit(filters=1024, kernel_size=(1, 1), strides=(2, 2))(shortcut) 



        x = Add()([x, shortcut]) # Skip Connection

        x = Activation(activation)(x)



    return x



# ResNet v2 fifth layer

# (conv4: 1x1_512(stride2 at i==0) -> 3x3_512 -> 1x1_2048)

def get_resnet_v2_conv5_layer(x, activation, count):

    for i in range(count):

        shortcut = x # for skip connection(스킵 연결)

        

        # output size 축소(가로 세로 절반으로)

        if (i == 0):

            x = resnet_conv2d_unit(filters=512, kernel_size=(1, 1), strides=(2, 2))(x)

        else:

            x = resnet_conv2d_unit(filters=512, kernel_size=(1, 1), strides=(1, 1))(x)



        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=512, kernel_size=(3, 3), strides=(1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)



        x = resnet_conv2d_unit(filters=2048, kernel_size=(1, 1), strides=(1, 1))(x)

        x = BatchNormalization()(x)



        # 처음 Skip Connection 할 때에는 출력 텐서의 차원과 shortcut 텐서의 차원이 서로 틀리므로 같게 만들어준다.

        # output size도 다르므로 stride를 2로 설정해서 output size의 크기를 1/4로 줄인다(가로와 세로가 둘 다 반씩 줄었으므로)

        if (i == 0):

            shortcut = resnet_conv2d_unit(filters=2048, kernel_size=(1, 1), strides=(2, 2))(shortcut) 



        x = Add()([x, shortcut]) # Skip Connection

        x = Activation(activation)(x)



    return x
# ResNet 구현

# learning rate reduction도 적용하기

def get_resnet_v2_50_layer_model():

    input_tensor = Input(shape=INPUT_SHAPE, dtype='float32')



    layer = get_resnet_v2_conv1_layer(input_tensor, activation=swish)

    layer = get_resnet_v2_conv2_layer(layer, activation=swish, count=3)

    layer = get_resnet_v2_conv3_layer(layer, activation=swish, count=4)

    layer = get_resnet_v2_conv4_layer(layer, activation=swish, count=6)

    layer = get_resnet_v2_conv5_layer(layer, activation=swish, count=3)



    layer = GlobalAveragePooling2D()(layer)



    # !! 시행 착오 !!

    # * 다른 곳에서는 초기값을 설정할 때 he 방법을 사용했는데, 마지막 output network의 가중치를 초기화 할 때

    #   kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)를 사용해주지 않았다.

    #   가중치를 검증된 방법을 초기화 하니까 훨씬 더 수렴이 잘 되었다.



    # grapheme_root

    output_grapheme_root       = Dense(GRAPHEME_ROOT_NUM,

                                       name='grapheme_root',

                                       activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(layer)

    # vowel_diacritic

    output_vowel_diacritic     = Dense(VOWEL_DIACRITIC_NUM,

                                       name='vowel_diacritic',

                                       activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(layer)

    # consonant_diacritic

    output_consonant_diacritic = Dense(CONSONANT_DIACRITIC_NUM,

                                       name='consonant_diacritic',

                                       activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(layer)



    return Model(inputs=input_tensor,

                 outputs=[output_grapheme_root, output_vowel_diacritic, output_consonant_diacritic])
# VGGNet 구현

def get_vggnet_model():

    activation = swish

    input_tensor = Input(shape=INPUT_SHAPE, dtype='float32')

    

    # input layer

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=activation)(input_tensor)



    # first block

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model)

    model = BatchNormalization(momentum=0.15)(model)



    # second block

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model)

    model = BatchNormalization(momentum=0.15)(model)

    model = Dropout(rate=0.3)(model)



    # third block

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model)

    model = BatchNormalization(momentum=0.15)(model)

    model = Dropout(rate=0.3)(model)



    # forth block

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model)

    model = BatchNormalization(momentum=0.15)(model)



    # fifth block

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation)(model)

    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model)

    model = Dropout(rate=0.3)(model)



    # fully connected

    model = Flatten()(model)

    model = Dense(2048, activation=activation)(model)

    model = Dropout(rate=0.2)(model)

    model = Dense(1024, activation=activation)(model)

    

    # classification block

    output_grapheme_root       = Dense(GRAPHEME_ROOT_NUM, name='grapheme_root', activation='softmax')(model)

    output_vowel_diacritic     = Dense(VOWEL_DIACRITIC_NUM, name='vowel_diacritic', activation='softmax')(model)

    output_consonant_diacritic = Dense(CONSONANT_DIACRITIC_NUM, name='consonant_diacritic', activation='softmax')(model)



    return Model(inputs=input_tensor,

                 outputs=[output_grapheme_root, output_vowel_diacritic, output_consonant_diacritic])
# Generalized mean pool - GeM

gm_exp = tf.Variable(3.0, dtype = tf.float32)

def generalized_mean_pool_2d(X):

    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 

                        axis = [1, 2], 

                        keepdims = False) + 1.e-7)**(1./gm_exp)

    return pool
# EfficientNet 만들기(외부 데이터 불러와서)

def get_efficiennet_model():

    activation = swish

    input_tensor = Input(shape=INPUT_SHAPE, dtype='float32')

    

    # Create and Compile Model and show Summary

    model = efn.EfficientNetB3(weights=None,

                               include_top=False,

                               input_tensor=input_tensor,

                               pooling=None,

                               classes=None)

    

    # UnFreeze all layers

    for layer in model.layers:

        layer.trainable = True

    

    # GeM

    lambda_layer = Lambda(generalized_mean_pool_2d)

    lambda_layer.trainable_weights.extend([gm_exp])

    x = lambda_layer(model.output)

    

    # multi output

    output_grapheme_root       = Dense(GRAPHEME_ROOT_NUM, name='grapheme_root', activation='softmax')(x)

    output_vowel_diacritic     = Dense(VOWEL_DIACRITIC_NUM, name='vowel_diacritic', activation='softmax')(x)

    output_consonant_diacritic = Dense(CONSONANT_DIACRITIC_NUM, name='consonant_diacritic', activation='softmax')(x)



    return Model(inputs=input_tensor,

                 outputs=[output_grapheme_root, output_vowel_diacritic, output_consonant_diacritic])
# model = get_resnet_v1_model(layer_num=(2, 2, 2, 2)) # 18-layer

# model = get_resnet_v1_model(layer_num=(3, 4, 6, 3)) # 34-layer

# model = get_resnet_v2_50_layer_model() # 54-layer

model = get_efficiennet_model() # efficient-net

print('total', model.count_params(), 'parameter(s)')



# optimizer = SGD(lr=0.01, momentum=0.9) # Stochastic gradient descent

optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
plot_model(model, to_file='model.png')
batch_size = 256

epochs = 12

validation_split = 0.2



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.001,

                              verbose=1, mode='auto')

callbacks = [reduce_lr]



def start_train(train_x, train_y):

    hist = model.fit(train_x, train_y,

                     batch_size=batch_size,

                     epochs=epochs,

                     validation_split=validation_split,

                     callbacks=callbacks,

                    )



def start_train_with_augmentation(train_x, train_y):   

    # 데이터 증강(실시간으로 처리) 요소

    datagen = ImageDataGenerator(

        rotation_range=10, # -10~10도 회전

        zoom_range=0.3     # 0.7~1.3배 확대

    )

    

    # Fit the model

    history = model.fit_generator(datagen.flow(train_x, {'grapheme_root': train_y[0], 'vowel_diacritic': train_y[1], 'consonant_diacritic': train_y[2]}, batch_size=batch_size),

                                  epochs = epochs, 

                                  steps_per_epoch=x_train.shape[0] // batch_size, 

                                  callbacks=callbacks)
# 학습 시작

print_log('train 0 start')

startTime = time.time()

start_train(train_x, train_y)

# start_train_with_augmentation(train_x, train_y)

del(train_x)

del(train_y)

gc.collect()

print_log('train 0 end')



for i in range(1, 4):

    print_log('train ' + str(i) + ' start')

    _train_x, _train_y = load_train_data_v2(TRAIN_IMG_PATH[i], output_img_size=(IMG_ROWS, IMG_COLUMNS), split_y=True)

    start_train(_train_x, _train_y)

#     start_train_with_augmentation(_train_x, _train_y)

    del(_train_x)

    del(_train_y)

    gc.collect()

    print_log('train ' + str(i) + ' end')



print('train elapsed time:', time.time() - startTime)
model.save('efficientnet_twotimes.h5')
# test_image_data.parquet 파일 경로

TEST_IMG_PATH = ['/kaggle/input/bengaliai-cv19/test_image_data_0.parquet',

           '/kaggle/input/bengaliai-cv19/test_image_data_1.parquet',

           '/kaggle/input/bengaliai-cv19/test_image_data_2.parquet',

           '/kaggle/input/bengaliai-cv19/test_image_data_3.parquet']
# train_image_data_x.parquet을 불러와서 numpy 배열로 되어 있는 이미지와, parquet 파일에 있는 Image_Id를 반환한다.

def get_img_data_with_img_id(img_path):

    data = pd.read_parquet(img_path)

    

    img_id = np.empty(shape=data.shape[0], dtype='object')

    img_id = data['image_id']

    

    # pandas로 데이터를 불러와 numpy 배열로 바꾸는 작업이 필요하다.

    # pandas로 데이터를 조작할 때에는 속도가 너무 느리다(특히 iloc). 따라서 numpy 배열로 바꾸어 반환해서 속도 향상을 꾀한다.

    img_data = data[data.columns[1:]]

    img_data = img_data.to_numpy(dtype='uint8')

    img_data = img_data.reshape(-1, RAW_IMG_ROWS, RAW_IMG_COLUMNS)

    

    del data

    gc.collect()

    

    return img_data, img_id
# 테스트 데이터 불러오기 Version 1

def load_test_data_v1(test_img_path):

    img_data, start_id, element_num = get_img_data(test_img_path)

    indices = np.array(range(start_id, start_id + element_num), dtype='uint32')



    # create test_x

    img_rows = RAW_IMG_ROWS // 2

    img_columns = RAW_IMG_COLUMNS // 2

    test_x = np.empty(dtype='uint8', shape=(element_num, img_rows, img_columns))

    

    for i in range(test_x.shape[0]): # resize image

        test_x[i] = cv2.resize(img_data[i], dsize=(img_columns, img_rows)) # width * height

    

    del img_data

    gc.collect()



    # keras CNN 모델에 넣기 위한 차원 및 데이터타입 정리

    test_x = test_x.reshape((-1, img_rows, img_columns, 1))

    test_x = test_x.astype('float32')

    test_x = test_x / 255.0



    return test_x, indices
# 테스트 데이터 불러오기 Version 2

def load_test_data_v2(test_img_path, output_img_size):

    img_data, img_id = get_img_data_with_img_id(test_img_path)

    

    element_num = img_data.shape[0]

    img_rows    = output_img_size[1]

    img_columns = output_img_size[0]

    

    # create test_x

    test_x = np.empty(dtype='uint8', shape=(element_num, img_rows, img_columns))

    for i in range(test_x.shape[0]):

        img = img_data[i]

        x1, y1, x2, y2 = get_img_roi(img) # get ROI(region of interest)

        img = img[y1:y2, x1:x2] # crop image

        img = cv2.resize(img, dsize=(img_columns, img_rows)) # resize image (width * height)

        test_x[i] = img

        

    # keras CNN 모델에 넣기 위한 차원 및 데이터타입 정리

    test_x = test_x.reshape((-1, img_rows, img_columns, 1))

    test_x = test_x.astype('float32')

    test_x = test_x / 255.0

        

    return test_x, img_id
def predict_model(y_split=True):

    # 리스트가 접근 속도가 빠름(pandas의 dataframe은 너무 느림)

    predict_result = list()

    

    for p in range(4):

        test_x, img_id = load_test_data_v2(TRAIN_IMG_PATH[p], output_img_size=(IMG_ROWS, IMG_COLUMNS))

        

        # 테스트 데이터 예측

        predict = model.predict(test_x)

        if y_split:

            predict_consonant_diacritic = predict[2].argmax(axis=1)

            predict_grapheme_root       = predict[0].argmax(axis=1)

            predict_vowel_diacritic     = predict[1].argmax(axis=1)

        else:

            predict_consonant_diacritic = predict[:, 179:].argmax(axis=1)

            predict_grapheme_root       = predict[:, :168].argmax(axis=1)

            predict_vowel_diacritic     = predict[:, 168:179].argmax(axis=1)



        # 예측값 기록

        for i in range(test_x.shape[0]):

            row_id_prefix = img_id[i] + '_'



            predict_result.append(row_id_prefix + 'consonant_diacritic')

            predict_result.append(predict_consonant_diacritic[i])



            predict_result.append(row_id_prefix + 'grapheme_root')

            predict_result.append(predict_grapheme_root[i])



            predict_result.append(row_id_prefix + 'vowel_diacritic')

            predict_result.append(predict_vowel_diacritic[i])

            

        print('predict test image data:', p)



    submission = pd.DataFrame(np.array(predict_result).reshape(-1, 2), columns=['row_id', 'target'])

    submission.set_index('row_id', inplace=True)

    

    return submission
# 시간 측정

startTime = time.time()



# 예측 시작

submission = predict_model(y_split=True)



print('predict elapsed time:', time.time() - startTime)
print(submission)

submission.to_csv('submission.csv')
test_x, img_id = load_test_data_v2(TEST_IMG_PATH[p], output_img_size=(IMG_ROWS, IMG_COLUMNS))