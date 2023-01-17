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

seed_everything(2020)

inputPath = '/kaggle/input/aiacademydeeplearning/train_images/'



image = cv2.imread(inputPath + '1_kitchen.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



display(image.shape)

display(image[0][0])



plt.figure(figsize=(8,4))

plt.imshow(image)
# 画像サイズの変更



image = cv2.resize(image, (128,128))

display(image.shape)

display(image[0][0])



plt.figure(figsize=(8,4))

plt.imshow(image)
# Targetデータを取得



train = pd.read_csv('/kaggle/input/aiacademydeeplearning/train.csv')

train = train.sort_values('id')

display(train.shape)

display(train.head())

display(train.dtypes)

display(train.isnull().sum())
# trainのデータの正規化



num_cols = ['bedrooms', 'bathrooms', 'area', 'zipcode']



scaler = StandardScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])



display(train.head())
# 画像を読み込み



def load_images(df, inputPath, size, rootType):

    images = []

    for i in df['id']:

        id_images = []

        basePath = os.path.sep.join([inputPath, "{}_{}*".format(i, roomType)])

        housePaths = sorted(list(glob.glob(basePath)))

        for housePath in housePaths:

            #print(housePath)

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            id_images.append(image)

        image = cv2.hconcat([cv2.vconcat([id_images[0], id_images[1]]), cv2.vconcat([id_images[2], id_images[3]]) ])

        #plt.figure(figsize=(8,4))

        #plt.imshow(image)

        images.append(image)

    return np.array(images) /255.0



# load train images



inputPath = '/kaggle/input/aiacademydeeplearning/train_images/'

size = 128

roomType = ['bathroom', 'bedroom','frontal', 'kitchen']

train_images = load_images(train, inputPath, size, roomType)

display(train_images.shape)

display(train_images[0][0][0])

# 訓練、検定データを作成



#train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2)



#train_y = train_x['price'].values

#valid_y = valid_x['price'].values



#display(train_images_x.shape)

#display(valid_images_x.shape)

#display(train_y.shape)

#display(valid_y.shape)
# CNNモデルを定義



def create_cnn(inputShape):

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid',

                    activation='relu',kernel_initializer='he_normal', input_shape=inputShape))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',

                    activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    #model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='valid',

    #                activation='relu', kernel_initializer='he_normal'))

    #model.add(MaxPooling2D(pool_size=(2,2)))

    #model.add(BatchNormalization())

    #model.add(Dropout(0.2))



    model.add(Flatten())

    

    #model.add(Dense(units=512, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(units=512, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(units=32, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    return model

# functional API



def create_fuctional(inputShape, num_cols):

    # 画像の処理

    image_input = Input(shape=(inputShape))

    x = Conv2D(16, (3,3), strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(image_input)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    

    x = Conv2D(16, (3,3), strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    

    image_output = Flatten()(x)

       

    # テーブルデータの処理

    table_input = Input(shape=(len(num_cols),))

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(table_input)

    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)

    table_output = Dense(1, activation='linear')(x)

    

    concatenated = layers.concatenate([image_output, table_output], axis=-1)



    x = Dense(512, activation='relu', kernel_initializer='he_normal')(concatenated)

    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

    output = Dense(1, activation='linear')(x)

    

    model = Model(inputs=[image_input, table_input], outputs=output)

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    

    return model



    

    
# モデル評価



def mean_absolute_percentage_error(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# 訓練の準備



# callback parameter

#filepath = "/kaggle/working/cnn_best_model.hdf5"

#es = EarlyStopping(patience=5, mode='min', verbose=1)

#checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto')

#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.01, verbose=1, mode='min')



# 訓練実行

#inputShape = (size, size, 3)



#model = create_cnn(inputShape)

#model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y), epochs=100, batch_size=16,

#         callbacks=[es, checkpoint, reduce_lr_loss])





# load best model weights

#model.load_weights(filepath)



# 評価

#valid_pred = model.predict(valid_images_x, batch_size=16).reshape((-1,1))

#mape_score = mean_absolute_percentage_error(valid_y, valid_pred)

#print(mape_score)


# 訓練実行（クロスバリデーション）

#splits = 4



#kf = KFold(n_splits=splits, shuffle=True, random_state=71)



#num_epochs = 80

#mape_scores = []

#all_mape_histories = []

#train_y = train['price'].values



#for train_index, val_index in kf.split(train_images, train_y):

#    train_data = train_images[train_index]

#    train_target = train_y[train_index]

#    val_data = train_images[val_index]

#    val_target = train_y[val_index]

    

    # callback parameter

    #filepath = "/kaggle/working/cnn_best_model.hdf5"

    #es = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1)

    #checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='min')

#    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.001, verbose=1, mode='min')

    

    # 訓練実行

#    inputShape = (size *2, size *2, 3)



#    model = create_cnn(inputShape)

#    history = model.fit(train_data, train_target, validation_data=(val_data, val_target), epochs=num_epochs, batch_size=16, callbacks=[es, checkpoint, reduce_lr_loss])

#    history = model.fit(train_data, train_target, validation_data=(val_data, val_target), epochs=num_epochs, batch_size=16, callbacks=[reduce_lr_loss])

    

    # load best model weights

    #model.load_weights(filepath)

    

    # 評価

    #valid_pred = model.predict(valid_images_x, batch_size=16).reshape((-1,1))

    #mape_scores.append(mean_absolute_percentage_error(valid_y, valid_pred))

    

#    mape_history = history.history['mape']

#    all_mape_histories.append(mape_history)



# 訓練実行（Fuctional APIのクロスバリデーション）

#splits = 4

#num_epochs = 50

#mape_scores = []

#all_mape_histories = []

#train_y = train['price'].values



#for i in range(71,73):

#    kf = KFold(n_splits=splits, shuffle=True, random_state=i)

#    for train_index, val_index in kf.split(train_images, train_y):

#        train_img_data = train_images[train_index]

#        train_data = train.query('index in @train_index')

#        train_target = train_y[train_index]

#        val_img_data = train_images[val_index]

#        val_data = train.query('index in @val_index')

#        val_target = train_y[val_index]

#        

#        # callback parameter

#        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.001, verbose=1, mode='min')

#        

#        # 訓練実行

#        inputShape = (size *2, size *2, 3)

#        

#        model = create_fuctional(inputShape, num_cols)

#        history = model.fit([train_img_data, train_data[num_cols]], train_target, validation_data=([val_img_data, val_data[num_cols]], val_target), epochs=num_epochs, batch_size=16, callbacks=[reduce_lr_loss])

#        

#        # 評価

#        mape_history = history.history['mape']

#        all_mape_histories.append(mape_history)

#print(np.mean(mape_scores))



#average_mape_history = [np.mean([x[i] for x in all_mape_histories]) for i in range(num_epochs)]



#plt.plot(range(1, len(average_mape_history)+1), average_mape_history)

#plt.xlabel('Epochs')

#plt.ylabel('Validation MAPE')

#plt.show()
model.summary()
plot_model(model, to_file='cnn.png')
# 全trainデータで再学習を実施



train_y = train['price'].values



# callback parameter

filepath = "/kaggle/working/cnn_best_model.hdf5"

es = EarlyStopping(monitor='mape', patience=7, mode='min', verbose=1)

checkpoint = ModelCheckpoint(monitor='mape', filepath=filepath, save_best_only=True, mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='mape', patience=2, factor=0.001, verbose=1, mode='min')



# 訓練実行

inputShape = (size *2, size *2, 3)



#model = create_cnn(inputShape)

model = model = create_fuctional(inputShape, num_cols)



model.fit([train_images, train[num_cols]], train_y, epochs=50, batch_size=16, callbacks=[es, checkpoint, reduce_lr_loss])

# 数値データの読み込み



test = pd.read_csv('/kaggle/input/aiacademydeeplearning/test.csv')

test = test.sort_values('id')



test[num_cols] = scaler.fit_transform(test[num_cols])



display(test.head())

display(test.shape)
# テスト画像の読み込み



inputPath = '/kaggle/input/aiacademydeeplearning/test_images/'

#size = 128

#roomType = 'frontal'

test_images = load_images(test, inputPath, size, roomType)

display(test_images.shape)

display(test_images[0][0][0])
# 予測を実行



# load best model weights

model.load_weights(filepath)

submit = model.predict([test_images, test[num_cols]], batch_size=16).reshape((-1,1))



#display(submit)



# sample_submission.csvファイルの読み込み



submission = pd.read_csv('/kaggle/input/aiacademydeeplearning/sample_submission.csv')

submission = submission.sort_values('id')

submission['price'] = 0
# submission用のDataFrameのターゲット列（price）に予測結果の値を代入



submission['price'] = submit



submission = submission.round(0)



display(submission)
# submission.csvファイルの書き出し



submission.to_csv('submission.csv', columns=['id','price'], index=False)