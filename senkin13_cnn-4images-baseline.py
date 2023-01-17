import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

%matplotlib inline



def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed = 817

seed_everything(seed)
inputPath = '/kaggle/input/aiacademydeeplearning/'

train = pd.read_csv(inputPath+'train.csv')

test = pd.read_csv(inputPath+'test.csv')

display(train.shape)

display(train.head())

display(test.shape)

display(test.head())
def load_images(df,inputPath,size):

    images = []

    for i in df['id']:

        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])

        housePaths = sorted(list(glob.glob(basePath)))

        inputImages = []

        outputImage = np.zeros((64, 64, 3))

        for housePath in housePaths:

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            inputImages.append(image)

        

        outputImage[0:32, 0:32] = inputImages[0]

        outputImage[0:32, 32:64] = inputImages[1]

        outputImage[32:64, 32:64] = inputImages[2]

        outputImage[32:64, 0:32] = inputImages[3]

        

        images.append(outputImage)

    

    return np.array(images) / 255.0



size = 32

# load train images

inputPath = '/kaggle/input/aiacademydeeplearning/train_images/'

train_images = load_images(train,inputPath,size)

display(train_images.shape)

display(train_images[0][0][0])

# load test images

inputPath = '/kaggle/input/aiacademydeeplearning/test_images/'

test_images = load_images(test,inputPath,size)

display(test_images.shape)

display(test_images[0][0][0])
def create_cnn(inputShape):

    model = Sequential()

  

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',

                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(Flatten())

    

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
def nn_kfold(train_df,test_df,train_images,test_images,imageShape,target,seed,network):

    n_splits= 5

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros((train_df.shape[0],1))

    sub_preds = np.zeros((test_df.shape[0],1))

    cv_list = []

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df)):



        train_x, train_y = train_images[train_idx], train_df[target].iloc[train_idx].values



        valid_x, valid_y = train_images[valid_idx], train_df[target].iloc[valid_idx].values

        

        test_x = test_images

        

        model = network(imageShape)

    

        filepath = str(n_fold) + "_nn_best_model.hdf5" 

        es = EarlyStopping(patience=5, mode='min', verbose=1) 

        checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_weights_only=True,mode='auto') 

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)



        hist = model.fit(train_x, train_y, batch_size=16, epochs=50, 

                         validation_data=(valid_x, valid_y), callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)



        model.load_weights(filepath)

        _oof_preds = model.predict(valid_x, batch_size=32,verbose=1)

        oof_preds[valid_idx] = _oof_preds.reshape((-1,1))



        oof_cv = mean_absolute_percentage_error(valid_y,  oof_preds[valid_idx])

        cv_list.append(oof_cv)

        print (cv_list)

        sub_preds += model.predict(test_x, batch_size=32).reshape((-1,1)) / folds.n_splits 

        

    cv = mean_absolute_percentage_error(train_df[target],  oof_preds)

    print('Full OOF MAPE %.6f' % cv)  



    train_df['prediction'] = oof_preds

    test_df['prediction'] = sub_preds    

    return train_df,test_df 
imageShape = (64, 64, 3) 

target = 'price'

network = create_cnn



train_nn,test_nn = nn_kfold(train,test,train_images,test_images,imageShape,target,seed,network)
train_nn.head()
test_nn.head()
test['price'] = test['prediction']

test_nn[['id','price']].to_csv('submission.csv',index=False)