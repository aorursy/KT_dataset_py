import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras import regularizers

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D

from tensorflow.keras.layers import Concatenate

from tensorflow.keras.optimizers import Adam, SGD

import pandas as pd
import glob

daisy = glob.glob('../input/flowers/Flowers/daisy/*.*')

dandelion = glob.glob('../input/flowers/Flowers/dandelion/*.*')

rose = glob.glob('../input/flowers/Flowers/rose/*.*')

sunflower = glob.glob('../input/flowers/Flowers/sunflower/*.*')

tulip = glob.glob('../input/flowers/Flowers/tulip/*.*')



data=[]

labels = []



for i in daisy:   

    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))

    image=np.array(image)

    data.append(image)

    labels.append(0)

for i in dandelion:   

    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))

    image=np.array(image)

    data.append(image)

    labels.append(1)

for i in rose:   

    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))

    image=np.array(image)

    data.append(image)

    labels.append(2)

for i in sunflower:   

    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))

    image=np.array(image)

    data.append(image)

    labels.append(3)

for i in tulip:   

    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))

    image=np.array(image)

    data.append(image)

    labels.append(4)



data = np.array(data)

labels = np.array(labels)
data.shape
from sklearn.model_selection import train_test_split

# Split into training set and testing set

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4,

                                                random_state=42)

# Split into validation set and testing set

X_valid,X_test,y_valid,y_test=train_test_split(X_test,y_test,test_size=0.5,random_state=42)



X_train=X_train/255

X_test=X_test/255

X_valid=X_valid/255

print("Training set: ",X_train.shape)

print("Validation set: ",X_valid.shape)

print("Testing set: ",X_test.shape)



# create model

def inception(x, filters):

    # 1x1

    path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)



    # 1x1->3x3

    path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)

    path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)

    

    # 1x1->5x5

    path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)

    path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)



    # 3x3->1x1

    path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)

    path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)



    return Concatenate(axis=-1)([path1,path2,path3,path4])





def auxiliary(x, name=None):

    layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)

    layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)

    layer = Flatten()(layer)

    layer = Dense(units=256, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(units=CLASS_NUM, activation='softmax', name=name)(layer)

    return layer





def googlenet():

    layer_in = Input(shape=IMAGE_SHAPE)

    

    # stage-1

    layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)

    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

    layer = BatchNormalization()(layer)



    # stage-2

    layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)

    layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)

    layer = BatchNormalization()(layer)

    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)



    # stage-3

    layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a

    layer = inception(layer, [128, (128,192), (32,96), 64]) #3b

    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

    

    # stage-4

    layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a

    aux1  = auxiliary(layer, name='aux1')

    layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b

    layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c

    layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d

    aux2  = auxiliary(layer, name='aux2')

    layer = inception(layer, [256, (160,320), (32,128), 128]) #4e

    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

    

    # stage-5

    layer = inception(layer, [256, (160,320), (32,128), 128]) #5a

    layer = inception(layer, [384, (192,384), (48,128), 128]) #5b

    layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)

    

    # stage-6

    layer = Flatten()(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(units=256, activation='linear',kernel_regularizer=regularizers.l2(0.0001))(layer)

    main = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)

    

    model = Model(inputs=layer_in, outputs=[main, aux1, aux2])

    

    return model
CLASS_NUM = 5

BATCH_SIZE = 16

EPOCH_STEPS = int(X_train.shape[0]/BATCH_SIZE)

IMAGE_SHAPE = (224, 224, 3)

MODEL_NAME = 'googlenet_flower.h5'
# train model

model = googlenet()

model.summary()

#model.load_weights(MODEL_NAME)

tf.keras.utils.plot_model(model, 'GoogLeNet.png')



optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#optimizer = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)

#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

optimizer = ['Adam', 'SGD', 'Adam', 'SGD']

epochs = [20, 30, 20, 30]

history_all = {}



history = model.fit(X_train,y_train,epochs=20,steps_per_epoch=EPOCH_STEPS,validation_data=(X_valid,y_valid))



model.save(MODEL_NAME)



score = model.evaluate(X_test, y_test)

print('Score:', score[4])

plt.plot(history.history['main_accuracy'])

plt.plot(history.history['val_main_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

plt.plot(history.history['main_loss'])

plt.plot(history.history['val_main_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()