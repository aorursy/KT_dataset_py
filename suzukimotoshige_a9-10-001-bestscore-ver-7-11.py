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
import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

import re

import gc

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV

from sklearn.preprocessing import StandardScaler



import tensorflow as tf

from keras import backend as K

#from tensorflow.keras import Input

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense, Input, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from tensorflow.keras.initializers import he_normal

from tensorflow.keras.regularizers import l2,l1

#from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input



import tqdm

import time

from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

import matplotlib.pyplot as plt

%matplotlib inline



###　Config

STEP = '011'



### kaggle・・・kaggle設定で上書き

MODE='kaggle'



#画像ファイルのディレクトリ

PATH_TO_TRAIN_IMAGE = 'train_images/'

PATH_TO_PRED_IMAGE = 'test_images/'

PATH_TO_TRAIN = 'train.csv'

PATH_TO_PRED = 'test.csv'

PATH_TO_SUBMIT= 'sample_submission.csv'

PATH_TO_MY_SUBMIT='my_submission_%s.csv' % (STEP)

PATH_TO_HDF5='cnn_best_model.hdf5'

#シード

RAND_SEED = 2020



#image size (256,256) 指定したサイズで正方形に変更しタイル状に加工。加工後はIMAGE_SIZE＊2

IMAGE_SIZE = 96



###テストデータの割合

PER_TEST = 0.2



#kaggle環境用設定

if MODE == 'kaggle':

    #画像ファイルのディレクトリ

    PATH_TO_TRAIN_IMAGE = '/kaggle/input/aiacademydeeplearning/train_images/'

    PATH_TO_PRED_IMAGE = '/kaggle/input/aiacademydeeplearning/test_images/'

    

    PATH_TO_TRAIN= '/kaggle/input/aiacademydeeplearning/train.csv'

    PATH_TO_PRED= '/kaggle/input/aiacademydeeplearning/test.csv'

    PATH_TO_SUBMIT= '/kaggle/input/aiacademydeeplearning/sample_submission.csv'

    #PATH_TO_MY_SUBMIT= '/kaggle/input/aiacademydeeplearning/' + PATH_TO_MY_SUBMIT

### 乱数シード

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

#========================================================

#

# 画像の読み込み

#

#========================================================

def load_images3(path):

    

    images=[] ## 結合後の画像リスト

    bedroom =[]

    bathroom=[]

    frontal    =[]

    kitchen   =[]

    pattern = re.compile(r'([0-9]{1,3})_(bathroom|bedroom|frontal|kitchen).jpg$')

    ### 画像ファイルのリスト取得＆ソート

    files=os.listdir(path=path)

    files.sort()

    

    for filename in files:

        res = pattern.match(filename)

        if res is None:

            continue

        

        #(id,image_type) = res.groups()

        ### 特定のIDを削除

        #if id in ['422']:

        #    print('image id:%s drop.%s' % (id,image_type))

        #    continue

            

        img = cv2.imread(path+filename)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))

        

        if 'bedroom'  in filename: bedroom.append(img)

        if 'bathroom' in filename: bathroom.append(img)

        if 'front'        in filename: frontal.append(img)

        if 'kitchen'    in filename: kitchen.append(img)

     

    ### 画像タイル結合

    for i in range(len(bedroom)):

        tiles = [[bedroom[i], bathroom[i]],[frontal[i], kitchen[i]]]

        image_concat = cv2.vconcat([cv2.hconcat(v_list) for v_list in tiles])

        images.append(image_concat)

               

    return np.array(images) / 255.0

def load_images(path):

    images=[] ## 結合後の画像リスト

    bedroom =[]

    bathroom=[]

    frontal    =[]

    kitchen   =[]

    pattern = re.compile(r'([0-9]{1,3})_(bathroom|bedroom|frontal|kitchen).jpg$')

    ### 画像ファイルのリスト取得＆ソート

    files=os.listdir(path=path)

    files.sort()

    

    for filename in files:

        res = pattern.match(filename)

        if res is None:

            continue

        

        #(id,image_type) = res.groups()

        ### 特定のIDを削除

        #if id in ['422']:

        #    print('image id:%s drop.%s' % (id,image_type))

        #    continue

        height = IMAGE_SIZE

        width = IMAGE_SIZE

        #if image_type in ['bedroom','bathroom']:

        #    height = int(IMAGE_SIZE*2 * 0.6)

        #else:

        #    height = IMAGE_SIZE*2 - int(IMAGE_SIZE*2 * 0.6)

            

        img = cv2.imread(path+filename)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img,(width,height))

        

        if 'bedroom'  in filename: bedroom.append(img)

        if 'bathroom' in filename: bathroom.append(img)

        if 'front'        in filename: frontal.append(img)

        if 'kitchen'    in filename: kitchen.append(img)

    

    ### 画像タイル結合

    for i in range(len(bedroom)):

        tiles = [[bedroom[i], bathroom[i]],[frontal[i], kitchen[i]]]

        image_concat = cv2.vconcat([cv2.hconcat(v_list) for v_list in tiles])

        images.append(image_concat)

        

    return np.array(images) / 255.0
#========================================================

#

# CNNモデルの定義

#

#========================================================

def create_cnn():

    model = Sequential()

    inputShape = (IMAGE_SIZE*2,IMAGE_SIZE*2,3)

    """

    演習:kernel_sizeを変更してみてください

    """

    #

    # 画像サイズに対してカーネルサイズが大きいとエラー。畳み込みを増やした場合

    #　適当に試してみる？？？3*3くらいから試す。

    #

    kernel_size=(5,5)

    #    stride　右に１、下に１

    #何次元のアウトプットにするか。filters

    #kernel_initializer=seed

    model.add(Conv2D(filters=32, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))

    # average pplingもあるけどmax poolingの方がよい

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))



    model.add(Conv2D(filters=64, kernel_size=(kernel_size), strides=(1, 1), padding='valid', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))

    

    model.add(Conv2D(filters=128, kernel_size=(2,2), strides=(1, 1), padding='valid', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))

    

    """

    演習:もう一層Conv2D->MaxPooling2D->BatchNormalization->Dropoutを追加してください

    """    

    #一次元配列に変換

    model.add(Flatten())

    

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
#========================================================

#

# CNNモデルの定義

#

#========================================================

def create_cnn_functional(df_shape,filters=16,kernel_size=(2,2),pool_size=(2,2),dropout=(0.3),seed=2020):

    

    inputShape = (IMAGE_SIZE*2,IMAGE_SIZE*2,3)

    input_image = Input(shape=(inputShape))

    input_df    = Input(shape=(df_shape,))

    

    

    ###image

    # layer-1 image

    model_image = Conv2D(filters=filters, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape)(input_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    # layer-2 image

    model_image = Conv2D(filters=filters*2, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape)(input_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    # layer-3 image

    model_image = Conv2D(filters=filters*4, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape)(input_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    model_image = Flatten()(model_image)

    

    ###train

    model_train = Dense(units=256, activation='relu',kernel_initializer='he_normal',input_shape=(df_shape,))(input_df)

    model_train = Dropout(dropout)(model_train)

    model_train = Dense(units=128, activation='relu',kernel_initializer='he_normal')(model_train) 

    model_train = Dropout(dropout)(model_train)

    model_train = Dense(units=64, activation='relu',kernel_initializer='he_normal')(model_train) 

    model_train = Dropout(dropout)(model_train)  

    model_train = Dense(units=1, activation='linear')(model_train) 

    

    ### model concate

    concate = concatenate([model_image,model_train])

    model_merge = Dense(units=256, activation='relu',kernel_initializer='he_normal')(concate) 

    model_merge = Dense(units=64, activation='relu',kernel_initializer='he_normal')(model_merge) 

    model_merge = Dense(units=1, activation='relu')(model_merge) 

    

    model = Model(inputs=[input_image, input_df], outputs=model_merge)

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    

    return model
#========================================================

#

# CNNモデルの定義

#

#========================================================

def create_cnn_functional_1(df_shape,filters=16,kernel_size=(2,2),pool_size=(2,2),dropout=(0.3),seed=2020):

    

    inputShape = (IMAGE_SIZE*2,IMAGE_SIZE*2,3)

    input_image = Input(shape=(inputShape))

    input_df    = Input(shape=(df_shape,))

    

    

    ###image

    # layer-1 image

    model_image = Conv2D(filters=filters, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape,kernel_regularizer=l2(0.01))(input_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    # layer-2 image

    model_image = Conv2D(filters=filters*2, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape,kernel_regularizer=l2(0.01))(input_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    # layer-3 image

    model_image = Conv2D(filters=filters*4, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape,kernel_regularizer=l2(0.01))(input_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    model_image = Flatten()(model_image)

    

    ###train

    model_train = Dense(units=256, activation='relu',kernel_initializer='he_normal',input_shape=(df_shape,))(input_df)

    model_train = Dropout(dropout)(model_train)

    model_train = Dense(units=128, activation='relu',kernel_initializer='he_normal')(model_train) 

    model_train = Dropout(dropout)(model_train)

    model_train = Dense(units=64, activation='relu',kernel_initializer='he_normal')(model_train) 

    model_train = Dropout(dropout)(model_train)  

    model_train = Dense(units=1, activation='linear')(model_train) 

    

    ### model concate

    concate = concatenate([model_image,model_train])

    model_merge = Dense(units=256, activation='relu',kernel_initializer='he_normal')(concate) 

    model_merge = Dense(units=64, activation='relu',kernel_initializer='he_normal')(model_merge) 

    model_merge = Dense(units=1, activation='relu')(model_merge) 

    

    model = Model(inputs=[input_image, input_df], outputs=model_merge)

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    

    return model
#========================================================

#

# CNNモデルの定義

#

#========================================================

def create_cnn_functional_vgg16(df_shape,filters=16,kernel_size=(2,2),pool_size=(2,2),dropout=(0.3),seed=2020):

    

    inputShape = (IMAGE_SIZE*2,IMAGE_SIZE*2,3)

    input_image = Input(shape=(inputShape))

    input_df    = Input(shape=(df_shape,))

    

    ###vgg16

    vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=inputShape)

    

    model_vgg16 = GlobalAveragePooling2D()(vgg16_conv.output)

    model_vgg16 = Flatten()(model_vgg16)

    model_vgg16 = Dense(512,kernel_initializer='he_normal', activation='relu')(model_vgg16)

    model_vgg16 = Dropout(dropout)(model_vgg16)

    model_vgg16 = Dense(256,kernel_initializer='he_normal', activation='relu')(model_vgg16)

    print(inputShape)

    print(input_image)

    print(model_vgg16)

    #print(model_vgg16.input)

    #print(model_vgg16.output)

    

    ###image

    # layer-1 image

    model_image = Conv2D(filters=filters, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape,kernel_regularizer=l2(0.01))(model_vgg16)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    # layer-2 image

    model_image = Conv2D(filters=filters*2, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape,kernel_regularizer=l2(0.01))(model_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    # layer-3 image

    model_image = Conv2D(filters=filters*4, kernel_size=(kernel_size), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer=he_normal(seed=seed), input_shape=inputShape,kernel_regularizer=l2(0.01))(model_image)

    model_image = MaxPooling2D(pool_size=(pool_size))(model_image)

    model_image = BatchNormalization()(model_image)

    model_image = Dropout(dropout)(model_image)

    

    model_image = Flatten()(model_image)

    

    #model_image = Model(inputs=input_image,outputs=[model_image,model_vgg16.output])

    

    #model_image = Model(inputs=input_image,outputs=[model_image,model_vgg16])

    model_image = Model(inputs=input_image,outputs=[model_image,model_vgg16.input])

    

    ###train

    model_train = Dense(units=256, activation='relu',kernel_initializer='he_normal',input_shape=(df_shape,))(input_df)

    model_train = Dropout(dropout)(model_train)

    model_train = Dense(units=128, activation='relu',kernel_initializer='he_normal')(model_train) 

    model_train = Dropout(dropout)(model_train)

    model_train = Dense(units=64, activation='relu',kernel_initializer='he_normal')(model_train) 

    model_train = Dropout(dropout)(model_train)  

    model_train = Dense(units=1, activation='linear')(model_train) 

    

    ### model concate

    concate = concatenate([model_image,model_train])

    #concate = concatenate([model_vgg16,model_train])

    model_merge = Dense(units=256, activation='relu',kernel_initializer='he_normal')(concate) 

    model_merge = Dense(units=64, activation='relu',kernel_initializer='he_normal')(model_merge) 

    model_merge = Dense(units=1, activation='relu')(model_merge) 

    

    model = Model(inputs=[input_image, input_df], outputs=model_merge)

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    model.summary()

    return model
def kfold_v1(train_images, df_train_sc, train_y, df_pred_sc, pred_images):

    

    KFOLD=5

    num_epochs = 16

    scores=[]

    inputShape = (IMAGE_SIZE*2, IMAGE_SIZE*2, 3)



    # callback parameter

    filepath = "cnn_best_model.hdf5" 

    es = EarlyStopping(patience=2, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=PATH_TO_HDF5, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min', factor=0.001)

    predict = pd.DataFrame()



    skf = StratifiedKFold(n_splits=KFOLD, random_state=71, shuffle=True)

    for i,(train_ix,test_ix) in enumerate(skf.split(train_images, train_y)):

        print('fold:',i)

        

        timekeeper(timelog,memo='kfold %s :' % (i))

        

        train_x_, train_y_, df_train_ = train_images[train_ix],train_y[train_ix], df_train_sc[train_ix]

        valid_x_, valid_y_, df_valid_ = train_images[test_ix], train_y[test_ix], df_train_sc[test_ix]

        #print(train_y_)

        #display(df_train_)

        #print(valid_y_)

        #display(df_valid_)

        if i == 0:

            model = create_cnn_functional_1(df_train.shape[1]-1)

        else:

            model = create_cnn_functional(df_train.shape[1]-1)

        

        model.fit([train_x_, df_train_],train_y_,

                  validation_data=([valid_x_, df_valid_], valid_y_),epochs=num_epochs, batch_size=1, verbose=1,

                  callbacks=[es, checkpoint, reduce_lr_loss])





        model.load_weights(PATH_TO_HDF5)

        

        ### 検定スコア

        valid_pred = model.predict([valid_x_,df_valid_], batch_size=16).reshape((-1,1))

        for v in valid_pred:

            v[0] = int(Decimal(v[0]/100).quantize(Decimal(0), rounding=ROUND_HALF_UP)*100)



        mape_score = mean_absolute_percentage_error(valid_y_.reshape((-1,1)), valid_pred)

        scores.append(mape_score)

        

        ###　予測（foldごとの予測値を最後に平均して返す

        y_pred_ = model.predict([pred_images,df_pred_sc], batch_size=16).reshape((-1,1))

        predict = pd.concat([predict, pd.DataFrame(y_pred_)], axis=1)

        gc.collect()

    return predict,scores

                  
def kfold_vgg16(train_images, df_train_sc, train_y, df_pred_sc, pred_images):

    

    KFOLD=5

    num_epochs = 16

    scores=[]

    all_mape_scores=[]

    inputShape = (IMAGE_SIZE*2, IMAGE_SIZE*2, 3)



    # callback parameter

    filepath = "cnn_best_model.hdf5" 

    es = EarlyStopping(patience=3, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=PATH_TO_HDF5, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min', factor=0.001)

    predict = pd.DataFrame()



    skf = StratifiedKFold(n_splits=KFOLD, random_state=71, shuffle=True)

    for i,(train_ix,test_ix) in enumerate(skf.split(train_images, train_y)):

        print('fold:',i)



        timekeeper(timelog,memo='kfold %s :' % (i))



        train_x_, train_y_, df_train_ = train_images[train_ix],train_y[train_ix], df_train_sc[train_ix]

        valid_x_, valid_y_, df_valid_ = train_images[test_ix], train_y[test_ix], df_train_sc[test_ix]



        #model = create_cnn_functional(df_train.shape[1]-1)

        model = create_cnn_functional_vgg16(df_train.shape[1]-1)

        

        history = model.fit([train_x_, df_train_],train_y_,

                  validation_data=([valid_x_, df_valid_], valid_y_),epochs=num_epochs, batch_size=1, verbose=1,

                  callbacks=[es, checkpoint, reduce_lr_loss])



        ### スコア記録

        all_mape_scores.append(history.history['val_mape'])



        model.load_weights(PATH_TO_HDF5)



        ### 検定スコア

        valid_pred = model.predict([valid_x_,df_valid_], batch_size=16).reshape((-1,1))

        for v in valid_pred:

            v[0] = int(Decimal(v[0]/100).quantize(Decimal(0), rounding=ROUND_HALF_UP)*100)



        mape_score = mean_absolute_percentage_error(valid_y_.reshape((-1,1)), valid_pred)

        scores.append(mape_score)



        ###　予測（foldごとの予測値を最後に平均して返す

        y_pred_ = model.predict([pred_images,df_pred_sc], batch_size=16).reshape((-1,1))

        predict = pd.concat([predict, pd.DataFrame(y_pred_)], axis=1)

        gc.collect()

    return predict,scores

                  
def mape_graph(all_mape_scores,num_epochs):

    average_mae_history=[

        np.mean([x[i] for x in all_mape_scores]) for i in range(num_epochs)

    ]



    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)

    plt.xlabel('epochs')

    plt.ylabel('validation mape')

    plt.show()
def leave_one_out(train_image_x, train_y, valid_images_x, valid_y):

    # callback parameter

    es = EarlyStopping(patience=5, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=PATH_TO_HDF5, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



    # 訓練実行

    inputShape = (IMAGE_SIZE*2, IMAGE_SIZE*2, 3)

    model = create_cnn()

    #batchsizeは32より精度がよい

    model.fit(train_images_x,  train_y, 

              validation_data=(valid_images_x, valid_y),

              epochs=50, batch_size=16,

        callbacks=[es, checkpoint, reduce_lr_loss])

    

    return model
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def timekeeper(log=None,memo=''):

    now = time.time()

    

    if log is None:

        log = [now]

    else:

        log.append('%s : %s' % (memo,round(now - log[0],1)) )

        

    return log
timelog = timekeeper(memo='start')

# 乱数シード固定

seed_everything(RAND_SEED)
###テーブル出た読み込み

df_train = pd.read_csv(PATH_TO_TRAIN)



###422だけ削除

#df_train = df_train[df_train['id'] != 422]



df_pred = pd.read_csv(PATH_TO_PRED)

display(df_train.shape)

display(df_train.head())

display(df_pred.head())

#画像ファイルの読み込み

train_images = load_images(PATH_TO_TRAIN_IMAGE)

pred_images = load_images(PATH_TO_PRED_IMAGE)



display(train_images.shape)

#display(train_images[0][0][0])



timekeeper(timelog,memo='load image')
### 学習データ、テストデータを作成

#(train_x, valid_x, train_images_x, valid_images_x) = train_test_split(df_train, train_images, test_size=PER_TEST)



train_y = df_train['price'].values

#valid_y = valid_x['price'].values



#display(train_images_x.shape)

#display(valid_images_x.shape)

#display(train_y.shape)

#display(valid_y.shape)





### データの正規化

scaler = StandardScaler()

scaler.fit(df_train.drop('price',axis=1))

df_train_sc = scaler.transform(df_train.drop('price',axis=1))



scaler.fit(df_pred)

df_pred_sc = scaler.transform(df_pred)

#(all_scores,all_mape_histories) = (train_images_x, train_y, valid_images_x, valid_y)

##まずはleave_one_out

#model = leave_one_out(train_images_x, train_y, valid_images_x, valid_y)

predict,scores = kfold_v1(train_images,df_train_sc,train_y,df_pred_sc,pred_images)



timekeeper(timelog,memo='kfold done')
print(scores)
#predict=predict1.copy()

predict['average'] = predict.mean(axis=1)

display(predict)
#predict['average'].round().astype(int)

### 予測値をround

#predict['average'] = predict['average'].round().astype(int)



predict['average']=predict['average'].apply(lambda x: Decimal(x/100).quantize(Decimal(0), rounding=ROUND_HALF_UP)*100)

display(predict)
display(scores)

timekeeper(timelog,memo='done')

display(timelog)

submit = pd.read_csv(PATH_TO_SUBMIT)

submit.price = predict.average

submit.head()
submit.to_csv(PATH_TO_MY_SUBMIT,index=False)