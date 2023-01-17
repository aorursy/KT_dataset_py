import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold

from sklearn.preprocessing import StandardScaler

from category_encoders import OneHotEncoder

from tensorflow.keras.applications import VGG16,VGG19

import tensorflow as tf

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense,Input,concatenate

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

%matplotlib inline
rootPath = '../input/4th-datarobot-ai-academy-deep-learning/'

imagesPath = rootPath + 'images/'

trainImPath = imagesPath + 'train_images/'

testImPath = imagesPath + 'test_images/'



train = pd.read_csv(rootPath+'train.csv')

test = pd.read_csv(rootPath+'test.csv')

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

        

        image_h1 = cv2.hconcat([inputImages[0], inputImages[1]])

        image_h2 = cv2.hconcat([inputImages[2], inputImages[3]])

        outputImage = cv2.vconcat([image_h1, image_h2])

        images.append(outputImage)

        

        plt.imshow(outputImage)

    

    np_images = np.array(images) / 255.0

    

    return np_images
size = 112

# load train images

train_images = load_images(train,trainImPath,size)

display(train_images.shape)

display(train_images[0][0][0])
# load test images

test_images = load_images(test,testImPath,size)

display(test_images.shape)

display(test_images[0][0][0])
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)
def create_multi_modal(num_cols,imageShape):

    # numerical feature

    input_num = Input(shape=(len(num_cols),), name='numerical')

    n = input_num

    n = Dense(512,activation='relu',kernel_initializer='he_normal',)(n)

    n = Dropout(0.2)(n)

    

    input_images = Input(shape=imageShape)

    backbone = VGG16(weights='imagenet', include_top=False, input_tensor=input_images, input_shape=imageShape)



    for layer in backbone.layers:

        print("{}: {}".format(layer, layer.trainable))

        

    # add a global spatial average pooling layer

    i = backbone.output

    i = GlobalAveragePooling2D()(i)    

#     i = Dense(1024, activation='relu', kernel_initializer='he_normal')(i)

#     i = Dropout(0.2)(i)

# #     i = Dense(512, activation='relu', kernel_initializer='he_normal')(i)

# #     i = Dropout(0.2)(i)    

#     i = Flatten()(i)

    

#     # image Layers

#     input_images = Input(shape=imageShape)

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

    i = Conv2D(256, (3, 3), padding="valid",activation='relu',kernel_initializer='he_normal')(i)

    i = MaxPooling2D(pool_size=(2, 2))(i)

    i = BatchNormalization()(i)

    i = Dropout(0.2)(i)    

    i = Flatten()(i)

    

    # multi input

    m = concatenate([n,i])

    m = Dense(256,activation='relu',kernel_initializer='he_normal',)(m)

    m = Dropout(0.2)(m)

    m = Dense(128,activation='relu',kernel_initializer='he_normal',)(m)

    m = Dropout(0.2)(m)

    m = Dense(32,activation='relu',kernel_initializer='he_normal',)(m)

    m = Dropout(0.2)(m)



    # check to see if the regression node should be added

    output = Dense(1)(m)



    model = Model([input_num]+[input_images], output)

    model.compile(loss="mape",optimizer='adam', metrics=['mape'])



    plot_model(model, to_file='multi_modal.png')

    return model
def nn_kfold(train_df,test_df,train_images,test_images,feats,imageShape,target,seed,network):

    seed_everything(seed)

    print('seed:' + str(seed))

    print('feats:' + str(train_df[feats].shape[1]))

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
# Feature Engineering of Table

def tableFE(train_df, test_df, feats):

    # 欠測値処理

    train_df[feats]=train_df[feats].fillna(0)

    test_df[feats]=test_df[feats].fillna(0)

    # Standard Scaler

    Scaler=StandardScaler()

    train_df[feats]=Scaler.fit_transform(train_df[feats])

    test_df[feats]=Scaler.fit_transform(test_df[feats])

    

    return train_df, test_df
size2 = size * 2

imageShape = (size2, size2, 3) 



feats = ['bedrooms', 'bathrooms', 'area', 'zipcode']

target = 'price'

train, test = tableFE(train, test, feats)

network = create_multi_modal



# 複数乱数シードmerge

for seed in [9,42,71,81,123,206,326,512,777,999]:

# for seed in [9,42,71,81,999]:

    train['prediction_' + str(seed)],test['prediction_' + str(seed)] = nn_kfold(train,test,train_images,test_images,feats,imageShape,target,seed,network)
train['prediction'] = (train['prediction_9'] + train['prediction_42'] + train['prediction_71'] + train['prediction_81'] + train['prediction_123']

                      + train['prediction_206'] + train['prediction_326'] + train['prediction_512'] + train['prediction_777'] + train['prediction_999'])/10

cv = mean_absolute_percentage_error(train[target],  train['prediction'].round(-2))

print('ALL SEED MAPE %.6f' % cv) 
test['prediction'] = (test['prediction_9'] + test['prediction_42'] + test['prediction_71'] + test['prediction_81'] + test['prediction_123']

                      + test['prediction_206'] + test['prediction_326'] + test['prediction_512'] + test['prediction_777'] + test['prediction_999'])/10

test['price'] = test['prediction'].round(-2)

test[['id','price']].to_csv('submission.csv',index=False)