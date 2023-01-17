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

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler  

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed_everything(2020)
train = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/train.csv')

test = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/test.csv')
train['rooms'] = train['bedrooms']+train['bathrooms']

test['rooms'] = test['bedrooms']+test['bathrooms']
# 特徴量

num_cols = ['bedrooms','bathrooms','area','zipcode','rooms']

target = ['price']
train_y = train[target]
scaler = StandardScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])

test[num_cols] = scaler.fit_transform(test[num_cols])



train_x = pd.concat([train['id'],train[num_cols]], axis=1)

test_x = pd.concat([test['id'],test[num_cols]], axis=1)
train_x = train_x.drop(columns='id')

test_x = test_x.drop(columns='id')
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def mlp(num_cols):



    model = Sequential()

    model.add(Dense(units=512, input_shape = (num_cols,), 

                    kernel_initializer='he_normal',activation='relu'))    

    model.add(Dropout(0.5))

    model.add(Dense(units=256,  kernel_initializer='he_normal',activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(units=32, kernel_initializer='he_normal', activation='relu'))     

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(skf.split(train_x, train_y)):

    X_train_, y_train_ = train_x.values[train_ix], train_y.values[train_ix]

    X_val, y_val = train_x.values[test_ix], train_y.values[test_ix]

    

    filepath = "mlp_best_model.hdf5" 

    

    es = EarlyStopping(patience=3, mode='min', verbose=1) 



    checkpoint = ModelCheckpoint(monitor='val_loss',filepath=filepath, save_best_only=True, mode='auto') 



    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1, verbose=1, mode='min')



    model = mlp(train_x.shape[1])

    

    history = model.fit(X_train_, y_train_,validation_data=(X_val, y_val),batch_size=32, epochs=200,

                    callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)

    

    model.load_weights(filepath)



    # predict valid data

    valid_pred = model.predict(X_val, batch_size=32).reshape((-1,1))

    valid_score = mean_absolute_percentage_error(y_val,  valid_pred)

    print ('valid mape:',valid_score)

    scores.append(valid_score)
np.array(scores).mean()
model.fit(train_x, train_y)
pred_num = model.predict(test_x, batch_size=32).reshape((-1,1))
pred_num
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def load_images_1(df,inputPath, size = 128):

    images = []

    for i in df['id']:

        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])

        housePaths = sorted(list(glob.glob(basePath)))

        outputImage = np.zeros((size, size, 3), dtype="uint8")

        inputImages = []

        for housePath in housePaths:

            image = cv2.imread(housePath)

            image = cv2.resize(image, (size//2, size//2))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            inputImages.append(image)

        outputImage[0:64, 0:64] = inputImages[0]

        outputImage[0:64, 64:128] = inputImages[1]

        outputImage[64:128, 64:128] = inputImages[2]

        outputImage[64:128, 0:64] = inputImages[3]

        images.append(outputImage)

    return np.array(images) / 255.0
images = []

size=128

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

for i in train['id']:

    basePath = os.path.sep.join([inputPath, "{}_*".format(i)])

    housePaths = sorted(list(glob.glob(basePath)))

    outputImage = np.zeros((size, size, 3), dtype="uint8")

    inputImages = []

    for housePath in housePaths:

        image = cv2.imread(housePath)

        image = cv2.resize(image, (size//2, size//2))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputImages.append(image)

    outputImage[0:64, 0:64] = inputImages[0]

    outputImage[0:64, 64:128] = inputImages[1]

    outputImage[64:128, 64:128] = inputImages[2]

    outputImage[64:128, 0:64] = inputImages[3]

    images.append(outputImage)
def vgg16_finetuning(inputShape):

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=inputShape)

   

    for layer in backbone.layers:

        print("{}: {}".format(layer, layer.trainable))

        

    model = Sequential(layers=backbone.layers)     

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    model.summary()

    return model
# load train images



pred_ls=[]

scores = []



#load image train

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

size = 128

train_images = load_images_1(train,inputPath,size)

display(train_images.shape)

display(train_images[0][0][0])



#load image test

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'

size = 128

test_images = load_images_1(test,inputPath,size)

display(train_images.shape)

display(train_images[0][0][0])

scores = []

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, test_ix) in enumerate(skf.split(train_images, train_y)):

    X_train_, y_train_ = train_images[train_ix], train_y.values[train_ix]

    X_val, y_val = train_images[test_ix], train_y.values[test_ix]



    # callback parameter

    filepath = "cnn_best_model.hdf5" 

    es = EarlyStopping(patience=5, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



    # 訓練実行

    datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             rotation_range=90,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             )

    inputShape = (size, size, 3)

    batch_size = 32

    model = vgg16_finetuning(inputShape)

    datagen.fit(X_train_,augment=True)

    train_datagen = datagen.flow(X_train_, y_train_, batch_size=batch_size, shuffle=True)

    history = model.fit(train_datagen, validation_data=(X_val, y_val),

        steps_per_epoch=len(X_train_) / batch_size, epochs=30,                

        callbacks=[es, checkpoint, reduce_lr_loss])



    model.load_weights(filepath)



    valid_pred = model.predict(X_val, batch_size=32).reshape((-1,1))

    mape_score = mean_absolute_percentage_error(y_val, valid_pred)

    print (mape_score)

    scores.append(mape_score)
np.array(scores).mean()
model.fit(train_images, train_y)
pred_image = model.predict(test_images, batch_size=32).reshape((-1,1))
pred_sub = ((pred_image*0.6+pred_num*1.4)/2).round()
pred_sub
submission = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/sample_submission.csv',index_col=0)
submission['price'] = pred_sub
submission.to_csv('submission.csv')