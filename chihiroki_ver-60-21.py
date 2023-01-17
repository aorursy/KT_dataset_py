#Version61で評価をお願いします(実際にベストスコアをとったのはver21ですが、それをコピーしたものです)
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

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

from keras.layers.recurrent import LSTM

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from tqdm import tqdm

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed_everything(2020)
submission_sumple=pd.read_csv('/kaggle/input/aiacademydeeplearning/sample_submission.csv')
#数値データ

train = pd.read_csv('/kaggle/input/aiacademydeeplearning/train.csv')

num_cols=['bedrooms','bathrooms','area','zipcode']

target=['price']

#欠測値処理

train[num_cols]=train[num_cols].fillna(-99999)

#正規化

Scaler=StandardScaler()

train[num_cols]=Scaler.fit_transform(train[num_cols])





#train_f=train[['bedrooms','bathrooms','area','zipcode']]

#train_p=train[['price']]

#display(train_f.shape)

#display(train.head())

#train_f.values
#テストデータ読み込み

test = pd.read_csv('/kaggle/input/aiacademydeeplearning/test.csv')

#欠測値処理

test[num_cols]=test[num_cols].fillna(-99999)

#正規化

Scaler=StandardScaler()

test[num_cols]=Scaler.fit_transform(test[num_cols])

display(test.shape)

display(test.head())
#画像を読み込み

def load_images(df,inputPath,size,roomType1,roomType2,roomType3,roomType4):

    images = []

    for i in df['id']:

        basePath1 = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType1)])

        basePath2 = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType2)])

        basePath3 = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType3)])

        basePath4 = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType4)])

        housePaths1 = sorted(list(glob.glob(basePath1)))

        housePaths2 = sorted(list(glob.glob(basePath2)))

        housePaths3 = sorted(list(glob.glob(basePath3)))

        housePaths4 = sorted(list(glob.glob(basePath4)))

        for housePath1 in housePaths1:

            image1 = cv2.imread(housePath1)

            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            image1 = cv2.resize(image1, (size, size))

        for housePath2 in housePaths2:

            image2 = cv2.imread(housePath2)

            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            image2 = cv2.resize(image2, (size, size))

        for housePath3 in housePaths3:

            image3 = cv2.imread(housePath3)

            image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

            image3 = cv2.resize(image3, (size, size))

        for housePath4 in housePaths4:

            image4 = cv2.imread(housePath4)

            image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

            image4 = cv2.resize(image4, (size, size))

        image1_2= cv2.vconcat([image1, image2])

        image3_4= cv2.vconcat([image3, image4])

        image_all=cv2.hconcat([image1_2, image1_2]) 

        #print(image_all.shape)

        #print(image_all)

        images.append(image_all)

    return np.array(images) / 255.0



# load train images

inputPath = '/kaggle/input/aiacademydeeplearning/train_images/'

size = 28

roomType1 = 'kitchen'

roomType2 = 'bathroom'

roomType3 = 'bedroom'

roomType4 = 'frontal'



train_images = load_images(train,inputPath,size,roomType1,roomType2,roomType3,roomType4)

display(train_images.shape)

display(train_images[0][0][0])

print(train_images.shape[1])
inputPath_test = '/kaggle/input/aiacademydeeplearning/test_images/'

test_images = load_images(test,inputPath_test,size,roomType1,roomType2,roomType3,roomType4)

display(test_images.shape)

display(test_images[0][0][0])
#評価関数定義

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print(y_pred.shape)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import tensorflow as tf

if int(tf.__version__.split('.')[0]) >= 2:

    from tensorflow import keras

else:

    import keras

#CNNモデルを定義する



inputs = keras.layers.Input(shape=(size*2, size*2,3))

lay1=keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid',

                         activation='relu', kernel_initializer='he_normal')(inputs)



lay2=(MaxPooling2D(pool_size=(2, 2)))(lay1)

lay3=(BatchNormalization())(lay2)

lay4=(Dropout(0.1))(lay3)



lay5=(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', 

                   activation='relu', kernel_initializer='he_normal'))(lay4)

lay6=(MaxPooling2D(pool_size=(2, 2)))(lay5)

lay7=(BatchNormalization())(lay6)

lay8=(Dropout(0.1))(lay7)



lay9=(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='valid', 

                      activation='relu', kernel_initializer='he_normal'))(lay8)

lay10=(MaxPooling2D(pool_size=(2, 2)))(lay9)

lay11=(BatchNormalization())(lay10)

lay12=(Dropout(0.1))(lay11)

lay13=(Flatten())(lay12)

lay14=(Dense(units=256, activation='relu',kernel_initializer='he_normal'))(lay13)  

lay15=(Dense(units=32, activation='relu',kernel_initializer='he_normal'))(lay14)  





inputs_mlp = keras.layers.Input(shape=(4, ))

lay1_mlp=Dense(units=512, input_shape = (len(num_cols),), 

                    kernel_initializer='he_normal',activation='relu')(inputs_mlp)    

lay2_mlp=Dropout(0.2)(lay1_mlp)

lay3_mlp=Dense(units=256,  kernel_initializer='he_normal',activation='relu')(lay2_mlp)

lay4_mlp=Dropout(0.2)(lay3_mlp)

lay5_mlp=Dense(units=32,  kernel_initializer='he_normal',activation='relu')(lay4_mlp)





#lay5_mlp=Dense(units=32, kernel_initializer='he_normal', activation='relu')(lay4_mlp)     

#lay6_mlp=Dropout(0.2)(lay5_mlp)

#lay7_mlp=Dense(1, activation='linear')(lay6_mlp)





merged = keras.layers.concatenate([lay14, lay4_mlp])





#lay15=(Dense(units=32, activation='relu',kernel_initializer='he_normal'))(lay14)

mer_lay1=(Dense(units=8, activation='relu',kernel_initializer='he_normal'))(merged)



mer_lay2=(Dense(units=1, activation='linear'))(mer_lay1)

model= keras.Model(inputs=[inputs,inputs_mlp], outputs=mer_lay2)



model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

# callback parameter

filepath = "cnn_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



    # 訓練実行

#Kfoldのと同様のことを以下で行う

n=5

y_pred=np.zeros(len(test))

mape_scores=[]

for i in range(n):

    train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2,random_state=i*10)

    train_y = train_x['price'].values

    valid_y = valid_x['price'].values

    train_table, valid_table= train_test_split(train, test_size=0.2,random_state=i*10)

    # 特徴量とターゲット

    train_f,train_t = train_table[num_cols].values,train_table[target].values

    valid_f,valid_t = valid_table[num_cols].values,valid_table[target].values





    model.fit([train_images_x,train_f], train_y, validation_data=([valid_images_x,valid_f], valid_y),epochs=50, batch_size=16,

            callbacks=[es, checkpoint, reduce_lr_loss])

    

    #評価

    model.load_weights(filepath)

    valid_pred = model.predict([valid_images_x,valid_f], batch_size=32).reshape((-1,1))

    mape_score = mean_absolute_percentage_error(valid_y, valid_pred)

    mape_scores.append(mape_score)

       

    #予測、アンサンブルをとる!

    test_pred = model.predict([test_images,test[num_cols].values],batch_size=32).reshape((-1,1))

    y_pred+=test_pred.reshape([len(test), ])     

ykai=y_pred/n

print(mape_scores)
#モデル可視化

#model.summary()
final=ykai
df = pd.DataFrame(final,columns=['price'])
submission_sumple2=submission_sumple.drop(['price'],axis=1)

kai=pd.concat([submission_sumple2,df],axis=1)

kai.to_csv('submission.csv',index=False)
kai
#ykai
#ykai_mlp