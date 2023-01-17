import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

import gc

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold

import tensorflow as tf

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense, Input,concatenate

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from tensorflow.keras.applications import VGG16,VGG19

from tensorflow.keras.applications.inception_v3 import InceptionV3

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

seed = 228

seed_everything(seed)
inputPath = '../input/4th-datarobot-ai-academy-deep-learning/'

train = pd.read_csv(inputPath+'train.csv')

train['price_bin'] = pd.cut(train['price'], [2000, 20000, 200000,500000,1000000,2000000], labels=[1, 2, 3, 4, 5])

train['price_bin'] = train['price_bin'].astype('int')

test = pd.read_csv(inputPath+'test.csv')

display(train.shape)

display(train.head())

display(test.shape)

display(test.head())



df = pd.concat([train,test],axis=0)

df = pd.get_dummies(df, columns=['bedrooms'])

df = pd.get_dummies(df, columns=['bathrooms'])

df = pd.get_dummies(df, columns=['area'])

df = pd.get_dummies(df, columns=['zipcode'])



train = df[df['price'].notnull()]

test = df[df['price'].isnull()]

del df

gc.collect()

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
def create_multi_modal(num_cols,image_cols):

    # numerical feature

    input_num = Input(shape=(len(num_cols),))

    n = input_num 

    

    # image feature

    input_images = Input(shape=image_cols)

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=image_cols)



    for layer in backbone.layers[:15]:

        layer.trainable = False

        

    for layer in backbone.layers:

        print("{}: {}".format(layer, layer.trainable))

        

    i = backbone.output    

    i = GlobalAveragePooling2D()(i)

    i = Dense(1,activation='relu',kernel_initializer='he_normal')(i)

    

    # merge

    x = concatenate([n,i])



    # mlp

    x = Dense(512,activation='relu',kernel_initializer='he_normal',)(x)

    x = Dropout(0.2)(x)

    x = Dense(256,activation='relu',kernel_initializer='he_normal',)(x)

    x = Dropout(0.2)(x)

    x = Dense(32,activation='relu',kernel_initializer='he_normal',)(x)

    x = Dropout(0.2)(x)



    # output

    output = Dense(1)(x)



    model = Model([input_num]+[backbone.input], output)

    adam = Adam(lr=0.001)

    model.compile(loss="mape",optimizer= adam, metrics=['mape'])

    model.summary()

    return model
class DataGenerator(tf.keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, X_num, X_image, y, list_IDs, batch_size, shuffle=False, augment=False, labels=True): 

        self.X_num = X_num

        self.X_image = X_image

        self.y = y

        self.augment = augment

        self.list_IDs = list_IDs

        self.shuffle = shuffle

        self.batch_size = batch_size

        self.labels = labels

        self.on_epoch_end()

        

    def __len__(self):

        'Denotes the number of batches per epoch'

        ct = len(self.list_IDs) // self.batch_size

        ct += int((len(self.list_IDs) % self.batch_size)!=0)

        return ct



    def __getitem__(self, index):

        'Generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self.__data_generation(indexes)

        if self.labels: return X, y

        else: return X



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange( len(self.list_IDs) )

        if self.shuffle: np.random.shuffle(self.indexes)



    def __data_generation(self, indexes):

        'Generates data containing batch_size samples'        

        X_num = np.array([self.X_num[k] for k in indexes])

        X_image = np.array([self.X_image[k] for k in indexes])

        y = np.array([self.y[k] for k in indexes])

        

        if self.augment: X_image = self.__augment_batch(X_image)

        X = (X_num, X_image)

        return X, y

 

    def __random_transform(self, img):

        datagen = ImageDataGenerator(horizontal_flip=True,

                           vertical_flip=True,          

                          )

        img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])

        datagen.fit(img,augment=True)

        return next(datagen.flow(img,batch_size=1))

    

    def __augment_batch(self, img_batch):

        for i in range(img_batch.shape[0]):

            img_batch[i, ] = self.__random_transform(img_batch[i, ])

        return img_batch

    
def nn_kfold(train_df,test_df,train_images,test_images,feats,imageShape,target,seed,network):

    seed_everything(seed)

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

        checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True,mode='auto') 

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)



        # augmentation

        train_datagen = DataGenerator(train_x,train_images_x,train_y,train_idx,shuffle=True,augment=True,batch_size=32)

        valid_datagen = DataGenerator(valid_x,valid_images_x,valid_y,valid_idx,shuffle=False,augment=False,batch_size=32)

 

        history = model.fit(train_datagen, validation_data=valid_datagen,

                            epochs=200,

                            callbacks=[es, checkpoint, reduce_lr_loss])

        

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

    

    return oof_preds,sub_preds
imageShape = (128, 128, 3) 

target = 'price'

drop_features = ['id', 'price', 'price_bin']

feats = [f for f in train.columns if f not in drop_features]

network = create_multi_modal



# 複数乱数シードmerge

for seed in [9,42,228,817,999]:

    train['prediction_' + str(seed)],test['prediction_' + str(seed)] = nn_kfold(train,test,train_images,test_images,feats,imageShape,target,seed,network)
test['price'] = (test['prediction_9'] + test['prediction_42'] + test['prediction_228'] + test['prediction_817'] + test['prediction_999'])/5

test[['id','price']].to_csv('submission.csv',index=False)