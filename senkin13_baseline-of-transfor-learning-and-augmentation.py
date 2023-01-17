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

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from tensorflow.keras.applications import VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
inputPath = '../input/4th-datarobot-ai-academy-deep-learning/'

train = pd.read_csv(inputPath+'train.csv')

train['price_bin'] = pd.cut(train['price'], [2000, 20000, 200000,500000,1000000,2000000], labels=[1, 2, 3,4,5])

train['price_bin'] = train['price_bin'].astype('int')

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

        outputImage = np.zeros((size*2, size*2, 3))

        for housePath in housePaths:

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            inputImages.append(image)

        

        outputImage[0:size, 0:size] = inputImages[0]

        outputImage[0:size, size:size*2] = inputImages[1]

        outputImage[size:size*2, size:size*2] = inputImages[2]

        outputImage[size:size*2, 0:size] = inputImages[3]

        

        images.append(outputImage)

    

    return np.array(images) / 255.0



size = 64

# load train images

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

train_images = load_images(train,inputPath,size)

display(train_images.shape)

display(train_images[0][0][0])

# load test images

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'

test_images = load_images(test,inputPath,size)

display(test_images.shape)

display(test_images[0][0][0])
def vgg16_finetuning(inputShape):

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=inputShape)

    

    for layer in backbone.layers[:15]:

        layer.trainable = False

        

    model = Sequential(layers=backbone.layers)     

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=512, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             #rotation_range=90,            

                             )
def nn_kfold(train_df,test_df,train_images,test_images,imageShape,target,seed,network):

    n_splits= 5

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros((train_df.shape[0],1))

    sub_preds = np.zeros((test_df.shape[0],1))

    cv_list = []

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['price_bin'])):



        train_x, train_y = train_images[train_idx], train_df[target].iloc[train_idx].values



        valid_x, valid_y = train_images[valid_idx], train_df[target].iloc[valid_idx].values

        

        test_x = test_images

        

        model = network(imageShape)

    

        filepath = str(n_fold) + "_nn_best_model.hdf5" 

        es = EarlyStopping(patience=8, mode='min', verbose=1) 

        checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True,mode='auto') 

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1)



        batch_size = 32

        datagen.fit(train_x,augment=True)

        train_datagen = datagen.flow(train_x, train_y, batch_size=batch_size, shuffle=True)

        history = model.fit(train_datagen, validation_data=(valid_x, valid_y),

                            steps_per_epoch=(len(train_x) / batch_size) , epochs=100,

                            callbacks=[es, checkpoint, reduce_lr_loss])

        

        model.load_weights(filepath)

        _oof_preds = model.predict(valid_x, batch_size=32,verbose=1)

        oof_preds[valid_idx] = _oof_preds.reshape((-1,1))



        oof_cv = mean_absolute_percentage_error(valid_y,  oof_preds[valid_idx])

        cv_list.append(oof_cv)

        print (cv_list)

        sub_preds += model.predict(test_x, batch_size=32).reshape((-1,1)) / folds.n_splits 

        

    cv = mean_absolute_percentage_error(train_df[target],  oof_preds)

    print('Full OOF MAPE %.6f' % cv)  



#     train_df['prediction'] = oof_preds

#     test_df['prediction'] = sub_preds    

    return oof_preds,sub_preds

imageShape = (128, 128, 3) 

target = 'price'

network = vgg16_finetuning



# 複数乱数シードmerge

for seed in [9,42,228,817,999]:

    train['prediction_' + str(seed)],test['prediction_' + str(seed)] = nn_kfold(train,test,train_images,test_images,imageShape,target,seed,network)

    

    
test['price'] = (test['prediction_9'] + test['prediction_42'] + test['prediction_228'] + test['prediction_817'] + test['prediction_999'])/5

test[['id','price']].to_csv('submission.csv',index=False)