import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras

from keras.models import Sequential, Model

from keras.layers import BatchNormalization,Activation,Dropout,Dense,Input

from keras.optimizers import Adam

from keras.utils import plot_model

from keras.layers import Flatten, Conv2D, MaxPooling2D, Embedding, GlobalAveragePooling2D,concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from keras import optimizers

from keras.applications.vgg16 import VGG16

import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed_everything(2020)
common_path='../input/4th-datarobot-ai-academy-deep-learning/'

train = pd.read_csv(common_path+'train.csv')

test = pd.read_csv(common_path+'test.csv')

submission = pd.read_csv(common_path+'sample_submission.csv')
train
test
df_train=train.copy()

df_test=test.copy()
def load_images(df,inputPath,size,count):

    images = []

    for num in range(count):

        for i in df['id']:

            basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'kitchen')])

            housePaths = sorted(list(glob.glob(basePath)))

            for housePath in housePaths:

                image_kitchen = cv2.imread(housePath)

                image_kitchen = cv2.cvtColor(image_kitchen, cv2.COLOR_BGR2RGB)

                image_kitchen = cv2.resize(image_kitchen, (size, size))

                image_kitchen = edit_img(image_kitchen,num)

                if num == 1:#trainデータのみ一枚水増し

                    image_kitchen = get_augmented(image_kitchen)



            

            basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'bathroom')])

            housePaths = sorted(list(glob.glob(basePath)))

            for housePath in housePaths:

                image_bathroom = cv2.imread(housePath)

                image_bathroom = cv2.cvtColor(image_bathroom, cv2.COLOR_BGR2RGB)

                image_bathroom = cv2.resize(image_bathroom, (size, size))

                image_bathroom = edit_img(image_bathroom,num)

                if num == 1:#trainデータのみ一枚水増し

                    image_bathroom = get_augmented(image_bathroom)

            

            basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'bedroom')])

            housePaths = sorted(list(glob.glob(basePath)))

            for housePath in housePaths:

                image_bedroom = cv2.imread(housePath)

                image_bedroom = cv2.cvtColor(image_bedroom, cv2.COLOR_BGR2RGB)

                image_bedroom = cv2.resize(image_bedroom, (size, size))

                image_bedroom = edit_img(image_bedroom,num)

                if num == 1:#trainデータのみ一枚水増し

                    image_bedroom = get_augmented(image_bedroom)



            basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'frontal')])

            housePaths = sorted(list(glob.glob(basePath)))

            for housePath in housePaths:

                image_frontal = cv2.imread(housePath)

                image_frontal = cv2.cvtColor(image_frontal, cv2.COLOR_BGR2RGB)

                image_frontal = cv2.resize(image_frontal, (size, size))

                image_frontal = edit_img(image_frontal,num)

                if num == 1:#trainデータのみ一枚水増し

                    image_frontal = get_augmented(image_frontal)

          

            image_m1 = cv2.vconcat([image_kitchen, image_bathroom])

            image_m2 = cv2.vconcat([image_bedroom, image_frontal])

            image = cv2.hconcat([image_m1, image_m2])

            image = cv2.resize(image, (size, size))

                         

            images.append(image)

        

    return np.array(images) / 255.0
def edit_img(image,num):

    if num == 1:

        image = np.fliplr(image)

    elif num == 2:

        imgsize = (size, size)

        center = (int(imgsize[0]/2), int(imgsize[1]/2))

        angle = -30

        scale = 1.0

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        image = cv2.warpAffine(image, rotation_matrix, imgsize, cv2.INTER_CUBIC)

    elif num == 3:

        imgsize = (size, size)

        center = (int(imgsize[0]/2), int(imgsize[1]/2))

        angle = -30

        scale = 1.0

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        image = cv2.warpAffine(image, rotation_matrix, imgsize, cv2.INTER_CUBIC)

        

    return image
def get_augmented(img, random_crop=4):



    if np.random.rand() > 0.5:

        img = np.fliplr(img)

        

    if np.random.rand() > 0.5:

        size = (img.shape[0], img.shape[1])

        center = (int(size[0]/2), int(size[1]/2))

        angle = np.random.randint(-30, 30)

        scale = 1.0

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        img = cv2.warpAffine(img, rotation_matrix, size, cv2.INTER_CUBIC)

    

    return img
#functional apiをつかってテーブルと画像データをMulti Inputでモデリング

#https://qiita.com/shu_marubo/items/eb297b2245040e50c00f



def mimodel(base_model,len_col):



    x = base_model.output

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.6)(x)

    x_output = Dense(32, activation='relu')(x)

    

    y_inputs = Input(shape=(len_col,))#カラム数変える場合注意

    y = Dense(512, activation='relu')(y_inputs)

    y = Dense(256, activation='relu')(y)

    y = Dropout(0.5)(y)   #0825追加

    y_output = Dense(8, activation='relu')(y)

    

    concat = concatenate([x_output,y_output])

    

    z = Dense(216, activation='relu')(concat)

    z = Dense(32, activation='relu')(z)

    z = Dropout(0.2)(z)   #0825追加

    prediction=Dense(units=1, activation='linear')(z)

    

    model=Model(inputs=[base_model.input,y_inputs],outputs=prediction)



    for layer in model.layers[:15]:

        layer.trainable=False



    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    

    return model
def seed_average(size,train_images,train_x,train_y,test_images,test_x,random_state,subm):

    inputShape = (size, size, 3)



    # 入力画像のサイズを指定

    input_tensor = Input(shape=inputShape)



    # 学習済みモデルの読み込み

    bottom_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    

    # callback parameter

    filepath = "cnn_best_model.hdf5"

    es = EarlyStopping(monitor='val_loss',patience=3, mode='min', verbose=1) #patience 初期値3

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')

    

    #cross validation

    #n_split：データの分割数．つまりk．検定はここで指定した数値の回数おこなわれる．

    #shuffle：Trueなら連続する数字でグループ分けせず，ランダムにデータを選択する．

    #random_state：乱数のシードを指定できる．

    split_num=5 #split回数

    kf = KFold(n_splits=split_num, shuffle=True, random_state=random_state)



    num=0

    subm2=subm.copy()

    bsize=random_state+1 #バッチサイズ  初期値32

    

    for train_index, val_index in kf.split(train_y):



        train_data=train_images[train_index]

        train_table=train_x.iloc[train_index]

        train_label=train_y[train_index]

        val_data=train_images[val_index]

        val_table=train_x.iloc[val_index]

        val_label=train_y[val_index]



        #訓練実行

        model = mimodel(bottom_model,len(train_x.columns))



        model.fit([train_data,train_table],

                  train_label,

                  validation_data=([val_data,val_table], val_label),

                  epochs=100,

                  batch_size=bsize,

                  callbacks=[es,reduce_lr_loss,checkpoint])



        model.load_weights(filepath)



        # 予測

        valid_test_pred = model.predict([test_images,test_x], batch_size=bsize).reshape((-1,1))

        subm['price'+str(num)]=valid_test_pred



        num += 1

        

    subm['price']=(subm['price0']+subm['price1']+subm['price2']+subm['price3']+subm['price4'])/5

    print('==============================')

    return subm['price']
## load train images

inputPath = common_path+'images/train_images/'

inputPath_test = common_path+'images/test_images/'

size = 224 #初期値224





train_images = load_images(df_train,inputPath,size,2)

test_images = load_images(df_test,inputPath_test,size,1)
train_images.shape
train = pd.concat([train,train])

train
#正解データ

train_y = train['price'].values
train_y
#学習データと予測用データを連結

data_all = pd.concat([train,test])
data_all
#郵便番号の頭にzをつける

data_all['zipcode'] = data_all['zipcode'].astype(str)

data_all['zipcode'] = 'z'+data_all['zipcode']



##郵便番号でonehot enc

df_zip = pd.get_dummies(data_all['zipcode'])

data_all = pd.concat([data_all,df_zip],axis=1)



#area(EDAで関連性なしと判断),zipcode,id,priceを削除

data_all = data_all.drop(["area","zipcode","id","price"],axis=1)
# 数値特徴量

num_cols = ['bedrooms', 'bathrooms']



# 正規化

scaler = StandardScaler()

data_all[num_cols] = scaler.fit_transform(data_all[num_cols])
#学習用とテスト用に分ける

train_onehot = data_all[:846]

test_onehot = data_all[846:]
train_onehot
test_onehot
submission_tmp = submission.copy()



num=0



randomseed=[0,4]

for n in randomseed:

    submission_tmp2=submission_tmp.copy()

    if num==0:

        submission_tmp['price']=seed_average(size,train_images,train_onehot,train_y,test_images,test_onehot,n,submission_tmp2)

    else:

        submission_tmp['price']+=seed_average(size,train_images,train_onehot,train_y,test_images,test_onehot,n,submission_tmp2)

    num += 1

   

submission['price'] =round(submission_tmp['price']/len(randomseed))
submission['price']=submission['price'].astype(np.int64)

submission
#submission

submission.to_csv("submission.csv",index=False)