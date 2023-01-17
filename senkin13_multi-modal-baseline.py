import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense,Input,concatenate

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
def create_multi_modal(num_cols,imageShape):

    # numerical feature

    input_num = Input(shape=(len(num_cols),), name='numerical')

    n = input_num

    n = Dense(512,activation='relu',kernel_initializer='he_normal',)(n)

    n = Dropout(0.2)(n)

    

    # Convolutionsl Layers

    input_images = Input(shape=imageShape)

    i = Conv2D(32, (3, 3), padding="valid",activation='relu',kernel_initializer='he_normal')(input_images)

    i = MaxPooling2D(pool_size=(2, 2))(i)

    i = BatchNormalization()(i)

    i = Dropout(0.2)(i)

    i = Conv2D(64, (3, 3), padding="valid",activation='relu',kernel_initializer='he_normal')(i)

    i = MaxPooling2D(pool_size=(2, 2))(i)

    i = BatchNormalization()(i)

    i = Dropout(0.2)(i)

    i = Conv2D(128, (3, 3), padding="valid",activation='relu',kernel_initializer='he_normal')(i)

    i = MaxPooling2D(pool_size=(2, 2))(i)

    i = BatchNormalization()(i)

    i = Dropout(0.2)(i)    

    i = Flatten()(i)

    

    # merge multi input

    x = concatenate([n,i])

    x = Dense(256,activation='relu',kernel_initializer='he_normal',)(x)

    x = Dropout(0.2)(x)

    x = Dense(128,activation='relu',kernel_initializer='he_normal',)(x)

    x = Dropout(0.2)(x)

    x = Dense(32,activation='relu',kernel_initializer='he_normal',)(x)

    x = Dropout(0.2)(x)



    # check to see if the regression node should be added

    output = Dense(1)(x)



    model = Model([input_num]+[input_images], output)

    model.compile(loss="mape",optimizer='adam', metrics=['mape'])



    plot_model(model, to_file='multi_modal.png')

    return model
def nn_kfold(train_df,test_df,train_images,test_images,feats,imageShape,target,seed,network):

    seed_everything(seed)

    print ('feats:' + str(train_df[feats].shape[1] ))

    n_splits= 5

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros((train_df.shape[0],1))

    sub_preds = np.zeros((test_df.shape[0],1))

    cv_list = []

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[target])):



        train_x, train_y, train_images_x = train_df[feats].iloc[train_idx].values, train_df[target].iloc[train_idx].values, train_images[train_idx]



        valid_x, valid_y, valid_images_x = train_df[feats].iloc[valid_idx].values, train_df[target].iloc[valid_idx].values, train_images[valid_idx]  

        

        test_x, test_images_x = test_df[feats].values, test_images



        model = network(feats,imageShape)

    

        filepath = str(n_fold) + "_nn_best_model.hdf5" 

        es = EarlyStopping(patience=5, mode='min', verbose=1) 

        checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_weights_only=True,mode='auto') 

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)



        hist = model.fit([train_x]+[train_images_x], train_y, batch_size=16, epochs=100, 

                         validation_data=([valid_x]+[valid_images_x], valid_y), callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)



        model.load_weights(filepath)

        _oof_preds = model.predict([valid_x]+[valid_images_x], batch_size=32,verbose=1)

        oof_preds[valid_idx] = _oof_preds.reshape((-1,1))



        oof_cv = mean_absolute_percentage_error(valid_y,  oof_preds[valid_idx])

        cv_list.append(oof_cv)

        print (cv_list)

        sub_preds += model.predict([test_x]+[test_images_x], batch_size=32).reshape((-1,1)) / folds.n_splits 

        

    cv = mean_absolute_percentage_error(train_df[target],  oof_preds)

    print('Full OOF MAPE %.6f' % cv)  



    train_df['prediction'] = oof_preds

    test_df['prediction'] = sub_preds    

    

    plot_model(model, to_file='multi_modal.png')

    return train_df['prediction'],test_df['prediction']  



def preprocess(train_df,test_df,feats):

    train_df = train_df.replace([np.inf, -np.inf], np.nan)

    train_df = train_df.fillna(0) 



    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    test_df = test_df.fillna(0)

    

    scaler = StandardScaler()

    train_df[feats] = scaler.fit_transform(train_df[feats])

    test_df[feats] = scaler.transform(test_df[feats])

    

    return train_df[feats], test_df[feats]
imageShape = (64, 64, 3) 

feats =  ['bedrooms','bathrooms']

target = 'price'

train[feats], test[feats] = preprocess(train,test,feats)

network = create_multi_modal



# 複数乱数シードmerge

for seed in [9,42,228,817,999]:

    train['prediction_' + str(seed)],test['prediction_' + str(seed)] = nn_kfold(train,test,train_images,test_images,feats,imageShape,target,seed,network)

    

train['prediction'] = (train['prediction_9'] + train['prediction_42'] + train['prediction_228'] + train['prediction_817'] + train['prediction_999'])/5

cv = mean_absolute_percentage_error(train[target],  train['prediction'])

print('ALL SEED MAPE %.6f' % cv) 
test['prediction'] = (test['prediction_9'] + test['prediction_42'] + test['prediction_228'] + test['prediction_817'] + test['prediction_999'])/5

test['price'] = test['prediction']

test[['id','price']].to_csv('submission.csv',index=False)