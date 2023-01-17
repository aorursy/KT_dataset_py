# !pip uninstall tensorflow -y

# !pip install tensorflow==1.14.0



# !pip uninstall keras -y

# !pip install keras==2.2.4
#tf.__version__
#keras.__version__
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
train = pd.read_csv('../input/aiacademydeeplearning/train.csv')

test = pd.read_csv("../input/aiacademydeeplearning/test.csv")

submission = pd.read_csv("../input/aiacademydeeplearning/sample_submission.csv")
##画像連結用

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

                #image_kitchen = edit_images(image_kitchen,num)

                if num == 1:

                    image_kitchen = get_augmented(image_kitchen)



            

            basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'bathroom')])

            housePaths = sorted(list(glob.glob(basePath)))

            for housePath in housePaths:

                image_bathroom = cv2.imread(housePath)

                image_bathroom = cv2.cvtColor(image_bathroom, cv2.COLOR_BGR2RGB)

                image_bathroom = cv2.resize(image_bathroom, (size, size))

                #image_bathroom = edit_images(image_bathroom,num)

                if num == 1:

                    image_bathroom = get_augmented(image_bathroom)

            

            basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'bedroom')])

            housePaths = sorted(list(glob.glob(basePath)))

            for housePath in housePaths:

                image_bedroom = cv2.imread(housePath)

                image_bedroom = cv2.cvtColor(image_bedroom, cv2.COLOR_BGR2RGB)

                image_bedroom = cv2.resize(image_bedroom, (size, size))

                #image_bedroom = edit_images(image_bedroom,num)

                if num == 1:

                    image_bedroom = get_augmented(image_bedroom)



            basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'frontal')])

            housePaths = sorted(list(glob.glob(basePath)))

            for housePath in housePaths:

                image_frontal = cv2.imread(housePath)

                image_frontal = cv2.cvtColor(image_frontal, cv2.COLOR_BGR2RGB)

                image_frontal = cv2.resize(image_frontal, (size, size))

                #image_frontal = edit_images(image_frontal,num)

                if num == 1:

                    image_frontal = get_augmented(image_frontal)

          

            image_m1 = cv2.vconcat([image_kitchen, image_bathroom])

            image_m2 = cv2.vconcat([image_bedroom, image_frontal])

            image = cv2.hconcat([image_m1, image_m2])

            image = cv2.resize(image, (size, size))

        

            # RGBからそれぞれvgg指定の値を引く

            #image[:, :, 0] = image[:, :, 0] - 100

            #image[:, :, 1] = image[:, :, 1] - 116.779

            #image[:, :, 2] = image[:, :, 2] - 123.68

                        

            images.append(image)

        

    return np.array(images) / 255.0



def edit_images(image,num):

    if num == 1:

        # 左右反転のノイズを加える

        image = np.fliplr(image)

    elif num == 2:

        #左に30度回転させる

        imgsize = (size, size)

        # 画像の中心位置(x, y)

        center = (int(imgsize[0]/2), int(imgsize[1]/2))

        # 回転させたい角度

        angle = -30

        # 拡大比率

        scale = 1.0

        # 回転変換行列の算出

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # 並進移動

        image = cv2.warpAffine(image, rotation_matrix, imgsize, cv2.INTER_CUBIC)

    elif num == 3:

        #右に30度回転させる

        imgsize = (size, size)

        # 画像の中心位置(x, y)

        center = (int(imgsize[0]/2), int(imgsize[1]/2))

        # 回転させたい角度

        angle = -30

        # 拡大比率

        scale = 1.0

        # 回転変換行列の算出

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # 並進移動

        image = cv2.warpAffine(image, rotation_matrix, imgsize, cv2.INTER_CUBIC)

        

    return image



def get_augmented(img, random_crop=4):



    # 左右反転のノイズを加える

    if np.random.rand() > 0.5:

        img = np.fliplr(img)

        

    # 左右どちらかに30度回転させる

    if np.random.rand() > 0.5:

        size = (img.shape[0], img.shape[1])

        # 画像の中心位置(x, y)

        center = (int(size[0]/2), int(size[1]/2))

        # 回転させたい角度

        angle = np.random.randint(-30, 30)

        # 拡大比率

        scale = 1.0

        # 回転変換行列の算出

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # 並進移動

        img = cv2.warpAffine(img, rotation_matrix, size, cv2.INTER_CUBIC)

    

    return img



# load train images

inputPath = '../input/aiacademydeeplearning/train_images/'

inputPath_test = '../input/aiacademydeeplearning/test_images/'

size = 224

train_images = load_images(train,inputPath,size,2)

test_images = load_images(test,inputPath_test,size,1)
train_images.shape
##アンサンブル用

# def load_images(df,inputPath,size,roomType):

#     images = []

#     for i in df['id']:

#         basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType)])

#         housePaths = sorted(list(glob.glob(basePath)))

#         for housePath in housePaths:

#             image = cv2.imread(housePath)

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             image = cv2.resize(image, (size, size))

#         images.append(image)

                

#     return np.array(images) / 255.0



# # load train images

# inputPath = '../input/aiacademydeeplearning/train_images/'

# inputPath_test = '../input/aiacademydeeplearning/test_images/'

# size = 224

# roomType = 'kitchen'

# train_images_kitchen = load_images(train,inputPath,size,roomType)

# test_images_kitchen = load_images(test,inputPath_test,size,roomType)

# roomType = 'bathroom'

# train_images_bathroom = load_images(train,inputPath,size,roomType)

# test_images_bathroom = load_images(test,inputPath_test,size,roomType)

# roomType = 'bedroom'

# train_images_bedroom = load_images(train,inputPath,size,roomType)

# test_images_bedroom = load_images(test,inputPath_test,size,roomType)

# roomType = 'frontal'

# train_images_frontal = load_images(train,inputPath,size,roomType)

# test_images_frontal = load_images(test,inputPath_test,size,roomType)
from keras.applications.vgg16 import VGG16



inputShape = (size, size, 3)



# 入力画像のサイズを指定

input_tensor = Input(shape=inputShape)



# 学習済みモデルの読み込み

base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
# x = base_model.output

# x = Flatten()(x)

# x = Dense(512, activation='relu')(x)

# x = Dropout(0.2)(x)

# x_output = Dense(32, activation='relu')(x)

    

# y_inputs = Input(shape=(4,))

# y = Dense(512, activation='relu')(y_inputs)

# y = Dropout(0.2)(y)

# y = Dense(32, activation='relu')(y)

# y_output = Dense(units=1, activation='linear')(y)

    

# concat = concatenate([x_output,y_output])

    

# z = Dense(216, activation='relu')(concat)

# z = Dense(32, activation='relu')(z)

# prediction=Dense(units=1, activation='linear')(z)

    

# model=Model(inputs=[base_model.input,y_inputs],outputs=prediction)



# for layer in model.layers[:19]:

#     layer.trainable=False



# model.compile(loss='mape', optimizer='adam', metrics=['mape'])
train = pd.concat([train,train])
#正解

train_y = train['price'].values
summary = pd.concat([train,test])
##テーブルデータ_train

#summary['area_label'] = 'a'

summary['zip_label'] = 'z'

#summary['area'] = summary['area'].astype(str)

summary['zipcode'] = summary['zipcode'].astype(str)

#summary['area'] = summary['area_label'].str.cat(summary['area'])

summary['zipcode'] = summary['zip_label'].str.cat(summary['zipcode'])

df_zip = pd.get_dummies(summary['zipcode'])

#df_area = pd.get_dummies(summary['area'])

summary = pd.concat([summary,df_zip],axis=1)

summary = summary.drop(["area","zipcode","zip_label","id","price"],axis=1)

# 特徴量

num_cols = ['bedrooms', 'bathrooms']



# 正規化

scaler = StandardScaler()

summary[num_cols] = scaler.fit_transform(summary[num_cols])
train_onehot = summary[:856]

test_onehot = summary[856:]
train_onehot
test_onehot
# ##テーブルデータ_test

# test['area_label'] = 'a'

# test['zip_label'] = 'z'

# test['area'] = test['area'].astype(str)

# test['zipcode'] = test['area'].astype(str)

# test['area'] = test['area_label'].str.cat(test['area'])

# test['zipcode'] = test['zip_label'].str.cat(test['zipcode'])

# df_zip = pd.get_dummies(test['zipcode'])

# df_area = pd.get_dummies(test['area'])

# test = pd.concat([test,df_zip,df_area],axis=1)

# test = test.drop(["area","zipcode","area_label","zip_label","id"],axis=1)

# # 特徴量

# num_cols = ['bedrooms', 'bathrooms']



# # 正規化

# scaler = StandardScaler()

# test[num_cols] = scaler.fit_transform(test[num_cols])
def conmodel():



    x = base_model.output

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.5)(x)

    x_output = Dense(32, activation='relu')(x)

    

    y_inputs = Input(shape=(51,))

    y = Dense(512, activation='relu')(y_inputs)

    y = Dense(256, activation='relu')(y)

    y_output = Dense(8, activation='relu')(y)

    

    concat = concatenate([x_output,y_output])

    

    z = Dense(216, activation='relu')(concat)

    #z = Dropout(0.5)(z)

    z = Dense(32, activation='relu')(z)

    #z = Dropout(0.5)(z)

    prediction=Dense(units=1, activation='linear')(z)

    

    #model=Model(inputs=base_model.input,outputs=prediction)

    model=Model(inputs=[base_model.input,y_inputs],outputs=prediction)



    for layer in model.layers[:15]:

        layer.trainable=False

        

    #optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    

    return model
# callback parameter

filepath = "cnn_best_model.hdf5" 

es = EarlyStopping(monitor='val_loss',patience=3, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')





submission_tmp = submission.copy()



kf = KFold(n_splits=5, shuffle=True)



num=0



for train_index, val_index in kf.split(train_y):



    train_data=train_images[train_index]

    train_table=train_onehot.iloc[train_index]

    train_label=train_y[train_index]

    val_data=train_images[val_index]

    val_table=train_onehot.iloc[val_index]

    val_label=train_y[val_index]

    

    model = conmodel()



    model.fit([train_data,train_table],

              train_label,

              validation_data=([val_data,val_table], val_label),

              epochs=100,

              batch_size=32,

              callbacks=[es,reduce_lr_loss,checkpoint])

    

    model.load_weights(filepath)

    

    # 予測

    valid_test_pred = model.predict([test_images,test_onehot], batch_size=32).reshape((-1,1))

    submission_tmp['price'+str(num)] = valid_test_pred

    

    num += 1
submission['price'] = (submission_tmp['price0'] + submission_tmp['price1'] + submission_tmp['price2'] + submission_tmp['price3'] + submission_tmp['price4'])/5

submission.to_csv("submission.csv",index=False)
#submission
##以下、アンサンブル用
##kitchen学習

# num=0

# submission_kitchen = submission.copy()



# for train_index, val_index in kf.split(train_y):



#     train_data_kitchen=train_images_kitchen[train_index]

#     #train_data_bathroom=train_images_bathroom[train_index]

#     #train_data_bedroom=train_images_bedroom[train_index]

#     #train_data_frontal=train_images_frontal[train_index]

#     #train_tbl=train_edit.iloc[train_index]

#     train_label=train_y[train_index]

#     val_data_kitchen=train_images_kitchen[val_index]

#     #val_data_bathroom=train_images_bathroom[val_index]

#     #val_data_bedroom=train_images_bedroom[val_index]

#     #val_data_frontal=train_images_frontal[val_index]   

#     val_label=train_y[val_index]

#     #val_tbl=train_edit.iloc[val_index]

    

#     model_kitchen = conmodel()



#     model_kitchen.fit(train_data_kitchen,

#               train_label,

#               validation_data=(val_data_kitchen, val_label),

#               epochs=30,

#               batch_size=16,

#               callbacks=[es,reduce_lr_loss])

    

#     # 予測

#     valid_test_pred = model_kitchen.predict(test_images_kitchen, batch_size=32).reshape((-1,1))

#     submission_kitchen['price'+str(num)] = valid_test_pred

    

#     num += 1

    

# submission_kitchen['price'] = (submission_kitchen['price0'] + submission_kitchen['price1'] + submission_kitchen['price2'] + submission_kitchen['price3'] + submission_kitchen['price4'])/5

# submission_kitchen = submission_kitchen.drop(['price0','price1','price2','price3','price4'],axis=1)

# submission_kitchen.to_csv("submission_kitchen.csv",index=False)
##bathroom学習

# num=0

# submission_bathroom = submission.copy()



# for train_index, val_index in kf.split(train_y):



#     #train_data_kitchen=train_images_kitchen[train_index]

#     train_data_bathroom=train_images_bathroom[train_index]

#     #train_data_bedroom=train_images_bedroom[train_index]

#     #train_data_frontal=train_images_frontal[train_index]

#     #train_tbl=train_edit.iloc[train_index]

#     train_label=train_y[train_index]

#     #val_data_kitchen=train_images_kitchen[val_index]

#     val_data_bathroom=train_images_bathroom[val_index]

#     #val_data_bedroom=train_images_bedroom[val_index]

#     #val_data_frontal=train_images_frontal[val_index]   

#     val_label=train_y[val_index]

#     #val_tbl=train_edit.iloc[val_index]

    

#     model_bathroom = conmodel()



#     model_bathroom.fit(train_data_bathroom,

#               train_label,

#               validation_data=(val_data_bathroom, val_label),

#               epochs=30,

#               batch_size=16,

#               callbacks=[es,reduce_lr_loss])

    

#     # 予測

#     valid_test_pred = model_bathroom.predict(test_images_bathroom, batch_size=32).reshape((-1,1))

#     submission_bathroom['price'+str(num)] = valid_test_pred

    

#     num += 1

    

# submission_bathroom['price'] = (submission_bathroom['price0'] + submission_bathroom['price1'] + submission_bathroom['price2'] + submission_bathroom['price3'] + submission_bathroom['price4'])/5

# submission_bathroom = submission_bathroom.drop(['price0','price1','price2','price3','price4'],axis=1)

# submission_bathroom.to_csv("submission_kitchen.csv",index=False)
##bedroom学習

# num=0

# submission_bedroom = submission.copy()



# for train_index, val_index in kf.split(train_y):



#     #train_data_kitchen=train_images_kitchen[train_index]

#     #train_data_bathroom=train_images_bathroom[train_index]

#     train_data_bedroom=train_images_bedroom[train_index]

#     #train_data_frontal=train_images_frontal[train_index]

#     #train_tbl=train_edit.iloc[train_index]

#     train_label=train_y[train_index]

#     #val_data_kitchen=train_images_kitchen[val_index]

#     #val_data_bathroom=train_images_bathroom[val_index]

#     val_data_bedroom=train_images_bedroom[val_index]

#     #val_data_frontal=train_images_frontal[val_index]   

#     val_label=train_y[val_index]

#     #val_tbl=train_edit.iloc[val_index]

    

#     model_bedroom = conmodel()



#     model_bedroom.fit(train_data_bedroom,

#               train_label,

#               validation_data=(val_data_bedroom, val_label),

#               epochs=30,

#               batch_size=16,

#               callbacks=[es,reduce_lr_loss])

    

#     # 予測

#     valid_test_pred = model_bedroom.predict(test_images_bedroom, batch_size=32).reshape((-1,1))

#     submission_bedroom['price'+str(num)] = valid_test_pred

    

#     num += 1

    

# submission_bedroom['price'] = (submission_bedroom['price0'] + submission_bedroom['price1'] + submission_bedroom['price2'] + submission_bedroom['price3'] + submission_bedroom['price4'])/5

# submission_bedroom = submission_bedroom.drop(['price0','price1','price2','price3','price4'],axis=1)

# submission_bedroom.to_csv("submission_bedroom.csv",index=False)
##frontal学習

# num=0

# submission_frontal = submission.copy()



# for train_index, val_index in kf.split(train_y):



#     #train_data_kitchen=train_images_kitchen[train_index]

#     #train_data_bathroom=train_images_bathroom[train_index]

#     #train_data_bedroom=train_images_bedroom[train_index]

#     train_data_frontal=train_images_frontal[train_index]

#     #train_tbl=train_edit.iloc[train_index]

#     train_label=train_y[train_index]

#     #val_data_kitchen=train_images_kitchen[val_index]

#     #val_data_bathroom=train_images_bathroom[val_index]

#     #val_data_bedroom=train_images_bedroom[val_index]

#     val_data_frontal=train_images_frontal[val_index]   

#     val_label=train_y[val_index]

#     #val_tbl=train_edit.iloc[val_index]

    

#     model_frontal = conmodel()



#     model_frontal.fit(train_data_frontal,

#               train_label,

#               validation_data=(val_data_frontal, val_label),

#               epochs=30,

#               batch_size=16,

#               callbacks=[es,reduce_lr_loss])

    

#     # 予測

#     valid_test_pred = model_frontal.predict(test_images_frontal, batch_size=32).reshape((-1,1))

#     submission_frontal['price'+str(num)] = valid_test_pred

    

#     num += 1

    

# submission_frontal['price'] = (submission_frontal['price0'] + submission_frontal['price1'] + submission_frontal['price2'] + submission_frontal['price3'] + submission_frontal['price4'])/5

# submission_frontal = submission_frontal.drop(['price0','price1','price2','price3','price4'],axis=1)

# submission_frontal.to_csv("submission_frontal.csv",index=False)
##アンサンブル

# submission['price'] = (submission_kitchen['price'] + submission_bathroom['price'] + submission_bedroom['price'] + submission_frontal['price'])/4

# submission.to_csv("submission.csv",index=False)