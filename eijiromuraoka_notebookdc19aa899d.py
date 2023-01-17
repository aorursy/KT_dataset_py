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





def load_images(df,inputPath,size,roomType):

    images = []

    for i in df['id']:

        basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType)])

        housePaths = sorted(list(glob.glob(basePath)))

        for housePath in housePaths:

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

        images.append(image)

    return np.array(images) / 255.0





# load train images

# 画像結合

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

inputPath1 = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'





size = 64

roomType = 'frontal'

train_images_frontal = load_images(train,inputPath,size,roomType)

test_images_frontal = load_images(test,inputPath1,size,roomType)





roomType = 'bathroom'

train_images_bathroom = load_images(train,inputPath,size,roomType)

test_images_bathroom = load_images(test,inputPath1,size,roomType)





roomType = 'bedroom'

train_images_bedroom = load_images(train,inputPath,size,roomType)

test_images_bedroom = load_images(test,inputPath1,size,roomType)





roomType = 'kitchen'

train_images_kitchen = load_images(train,inputPath,size,roomType)

test_images_kitchen = load_images(test,inputPath1,size,roomType)





train_images1 = np.concatenate([train_images_frontal,train_images_bathroom],1)

train_images2 = np.concatenate([train_images_bedroom,train_images_kitchen],1)

train_images =  np.concatenate([train_images1,train_images2],2)



test_images1 = np.concatenate([test_images_frontal,test_images_bathroom],1)

test_images2 = np.concatenate([test_images_bedroom,test_images_kitchen],1)

test_images =  np.concatenate([test_images1,test_images2],2)











train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2)

train_y = train_x['price'].values

valid_y = valid_x['price'].values

display(train_images_x.shape)

display(valid_images_x.shape)

display(train_y.shape)

display(valid_y.shape)





def create_cnn(inputShape):

    model = Sequential()

 

    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same',

                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))



    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))



#一層追加    

    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))

    

    

    model.add(Flatten())

    

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model





# callback parameter

filepath = "cnn_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

inputShape = (size*2, size*2, 3)

model = create_cnn(inputShape)

history = model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y),epochs=30, batch_size=32,

    callbacks=[es, checkpoint, reduce_lr_loss])









from tensorflow.keras.applications import VGG16



def vgg16_feature_extraction(inputShape):

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=inputShape)

    

    model = Sequential(layers=backbone.layers)     

    model.add(Flatten())  

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    model.trainable = False    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    model.summary()

    return model



# 訓練実行

inputShape = (size*2, size*2, 3)

model = vgg16_feature_extraction(inputShape)

history = model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y),epochs=30, batch_size=32,

    callbacks=[es, checkpoint, reduce_lr_loss])





def vgg16_finetuning(inputShape):

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=inputShape)

    """

    演習:Convolution Layerの重みを全部訓練してみてください！

    """    

    

    for layer in backbone.layers[:15]:

         layer.trainable = False

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





from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             )













# callback parameter

filepath = "transfer_learning_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

inputShape = (size*2, size*2, 3)

model = vgg16_finetuning(inputShape)

history = model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y),epochs=30, batch_size=32,

    callbacks=[es, checkpoint, reduce_lr_loss])



# load best model weights

model.load_weights(filepath)









# callback parameter

filepath = "transfer_learning_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

inputShape = (size, size, 3)

model = vgg16_finetuning(inputShape)

history = model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y),epochs=30, batch_size=32,

    callbacks=[es, checkpoint, reduce_lr_loss])



# load best model weights

model.load_weights(filepath)













def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# load best model weights

model.load_weights(filepath)



# 評価

valid_pred = model.predict(valid_images_x, batch_size=32).reshape((-1,1))

mape_score = mean_absolute_percentage_error(valid_y, valid_pred)

print (mape_score)







y_test=model.predict(test_images, batch_size=32).reshape((-1,1))

submission = pd.DataFrame({

"id": test.id,

"price": y_test.T[0]

})

submission.to_csv('submission.csv', index=False)
