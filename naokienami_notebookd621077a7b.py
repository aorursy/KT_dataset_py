# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime

import random

import glob

import cv2

import os

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from tensorflow.keras import layers

from tensorflow.keras import Input

import matplotlib.pyplot as plt

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed_everything(1234)

# トレーニングデータのロード

train = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/train.csv')

train = train.sort_values('id')



# データの概要把握

print('------------------------------------')

print(train.shape)

print('------------------------------------')

print(train.head())

print('------------------------------------')

print(train.dtypes)

print('------------------------------------')

print(train.isnull().sum())



# 交互作用項の追加

#train['bed_bath'] = train['bedrooms'] * train['bathrooms']
# trainのデータの正規化

# num_cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'bed_bath']

num_cols = ['bedrooms', 'bathrooms', 'area', 'zipcode']



scaler = StandardScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])



print(train.head())
# 画像を読み込む関数

def load_images(df, inputPath, size, roomType):

    images = []

    for i in df['id']:

        id_images = []

        basePath = os.path.sep.join([inputPath, "{}_{}*".format(i, roomType)])

        housePaths = sorted(list(glob.glob(basePath)))

        # 該当するIDのファイル名分だけ回す

        for housePath in housePaths:

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            id_images.append(image)

        # 4枚の画像を1枚に連結

        image = cv2.hconcat([cv2.vconcat([id_images[0], id_images[1]]), cv2.vconcat([id_images[2], id_images[3]]) ])

        images.append(image)

    return np.array(images) / 255.0



# TODO: 時間があれば水増し処理を加える
# 画像データの読み込み

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

size = 256

roomType = ['bathroom', 'bedroom','frontal', 'kitchen']

train_images = load_images(train, inputPath, size, roomType)



# [id分, size*2, size*2, 3]になるはず

print(train_images.shape)



# ****************時間があればデータの水増しを行う****************
# CNNモデルを定義

def create_cnn(inputShape):

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',

                    activation='relu',kernel_initializer='he_normal', input_shape=inputShape))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid',

                    activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='valid',

                   activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Flatten())

    

    model.add(Dense(units=512, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(units=256, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(units=32, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    return model

# functional API

def create_fuctional(inputShape, num_cols):

    # 画像を扱う中間層

    image_input = Input(shape=(inputShape))

    x = Conv2D(32, (5,5), strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(image_input)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    

    x = Conv2D(64, (5,5), strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    

    x = Conv2D(128, (5,5), strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)  



    x = Conv2D(256, (5,5), strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)  

    image_output = Flatten()(x)

       

    # テーブルデータを扱う中間層

    table_input = Input(shape=(len(num_cols),))

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(table_input)

    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.3)(x)

    table_output = Dense(1, activation='linear')(x)

    

    # 合流させる

    concatenated = layers.concatenate([image_output, table_output], axis=-1)



    # 全結合層

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(concatenated)

    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)

    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)

    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

    output = Dense(1, activation='linear')(x)

    

    model = Model(inputs=[image_input, table_input], outputs=output)

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    

    return model
# VGG16の活用

from keras.applications.vgg16 import VGG16

def create_VGG_fuctional(inputShape, num_cols):

    # 入力画像のサイズを指定

    image_input = Input(shape=(inputShape))



    # 学習済みモデルの読み込み

    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)

    

    # 学習可能にする

    base_model.trainable = True

    

    # 画像を扱う中間層

    x = base_model.output

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)

    x = Dropout(0.2)(x)

    image_output = Flatten()(x)   

    

    # テーブルデータを扱う中間層

    table_input = Input(shape=(len(num_cols),))

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(table_input)

    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)

    table_output = Dense(1, activation='linear')(x)

    

    # 合流させる

    concatenated = layers.concatenate([image_output, table_output], axis=-1)



    # 全結合層

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(concatenated)

    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)

    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

    output = Dense(1, activation='linear')(x)

    

    model = Model(inputs=[image_input, table_input], outputs=output)

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    

    return model
# 学習

train_y = train['price'].values



# callback parameter

filepath = "/kaggle/working/cnn_best_model.hdf5"

es = EarlyStopping(monitor='mape', patience=5, mode='min', verbose=1)

checkpoint = ModelCheckpoint(monitor='mape', filepath=filepath, save_best_only=True, mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='mape', patience=5, factor=0.001, verbose=1, mode='min')



# 訓練実行

inputShape = (size*2, size*2, 3)



#model = create_cnn(inputShape)

model = create_fuctional(inputShape, num_cols)

# model = create_VGG_fuctional(inputShape, num_cols)



model.fit([train_images, train[num_cols]], train_y, epochs=100, batch_size=8, callbacks=[es, checkpoint, reduce_lr_loss])

# テスト数値データの読み込み

test = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/test.csv')

test = test.sort_values('id')



# 交互作用項の追加

#test['bed_bath'] = test['bedrooms'] * test['bathrooms']



test[num_cols] = scaler.fit_transform(test[num_cols])



print(test.head())

print(test.shape)
# テスト画像の読み込み

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'



test_images = load_images(test, inputPath, size, roomType)



print(test_images.shape)
# 予測を実行

# load best model weights

model.load_weights(filepath)

submit = model.predict([test_images, test[num_cols]], batch_size=8).reshape((-1,1))
# sample_submission.csvファイルの読み込み

submission = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/sample_submission.csv')

submission = submission.sort_values('id')

submission['price'] = 0
# 予測結果の代入

submission['price'] = submit

submission = submission.round(0)



# submission.csvファイルの書き出し

submission.to_csv('submission.csv', columns=['id','price'], index=False)