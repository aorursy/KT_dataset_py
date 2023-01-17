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
inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

# 画像読み込み

image = cv2.imread(inputPath+'1_bathroom.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

display(image.shape)

display(image[0][0])

# 画像を表示

plt.figure(figsize=(8,4))

plt.imshow(image)
# 画像のサイズ変更

image = cv2.resize(image,(256,256))

display(image.shape)

display(image[0][0])

# 画像を表示

plt.figure(figsize=(8,4))

plt.imshow(image)
train = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/train.csv')

display(train.shape)

display(train.head())
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

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

size = 64

roomType = 'frontal'

train_images = load_images(train,inputPath,size,roomType)

display(train_images.shape)

display(train_images[0][0][0])
train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2)

train_y = train_x['price'].values

valid_y = valid_x['price'].values

display(train_images_x.shape)

display(valid_images_x.shape)

display(train_y.shape)

display(valid_y.shape)
def create_cnn(inputShape):

    model = Sequential()

    """

    演習:kernel_sizeを変更してみてください

    """    

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

    """

    演習:もう一層Conv2D->MaxPooling2D->BatchNormalization->Dropoutを追加してください

    """    

    

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

inputShape = (size, size, 3)

model = create_cnn(inputShape)

history = model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y),epochs=30, batch_size=32,

    callbacks=[es, checkpoint, reduce_lr_loss])

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# load best model weights

model.load_weights(filepath)



# 評価

valid_pred = model.predict(valid_images_x, batch_size=32).reshape((-1,1))

mape_score = mean_absolute_percentage_error(valid_y, valid_pred)

print (mape_score)
model.summary()
plot_model(model, to_file='cnn.png')
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'bo' ,label = 'training loss')

plt.plot(epochs, val_loss, 'b' , label= 'validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
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

inputShape = (size, size, 3)

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



# 評価

valid_pred = model.predict(valid_images_x, batch_size=32).reshape((-1,1))

mape_score = mean_absolute_percentage_error(valid_y, valid_pred)

print (mape_score)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             )



inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

image = cv2.imread(inputPath+'1_bathroom.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_data = image / 255.0

image_data = image_data.reshape(1,image_data.shape[0],image_data.shape[1],image_data.shape[2])



print (image_data.shape)

for batch in datagen.flow(image_data,batch_size=1):

    plt.imshow(image_data[0])

    plt.show() 

    plt.imshow(batch[0])

    plt.show()       

    break
"""

演習:ImageDataGenerator中に新たな三つを追加してみてください！

                             rotation_range=90,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

"""    



# callback parameter

filepath = "data_aug_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             )

inputShape = (size, size, 3)

batch_size = 32

model = vgg16_finetuning(inputShape)

datagen.fit(train_images_x,augment=True)

train_datagen = datagen.flow(train_images_x, train_y, batch_size=batch_size, shuffle=True)

history = model.fit(train_datagen, validation_data=(valid_images_x, valid_y),

    steps_per_epoch=len(train_images_x) / batch_size, epochs=30,                

    callbacks=[es, checkpoint, reduce_lr_loss])







# load best model weights

model.load_weights(filepath)



# 評価

valid_pred = model.predict(valid_images_x, batch_size=32).reshape((-1,1))

mape_score = mean_absolute_percentage_error(valid_y, valid_pred)

print (mape_score)