from keras.models import Model

from keras.layers import *

from keras.datasets import mnist

import numpy as np

import matplotlib.pyplot as plt

import keras

import tensorflow as tf

from keras.utils import to_categorical
batch_size = 256

epochs = 20

inChannel = 1

x, y = 28, 28

input_img = Input(shape = (x, y, inChannel))

num_classes = 10
def encoder(input_img):

    #encoder

    #input = 28 x 28 x 1 (wide and thin)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32

    conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64

    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)

    conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    conv4 = BatchNormalization()(conv4)

    return conv4



def decoder(conv4):    

    #decoder

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128

    conv5 = BatchNormalization()(conv5)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64

    conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    conv6 = BatchNormalization()(conv6)

    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64

    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32

    conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    conv7 = BatchNormalization()(conv7)

    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1

    return decoded
autoencoder = Model(input_img, decoder(encoder(input_img)))

autoencoder.compile(loss='mean_squared_error', optimizer = 'rmsprop')
autoencoder.summary()
import pandas as pd

import numpy as np

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
label = train['label']

train = train.drop('label',axis=1)
train = np.array(train)/255

train = train.reshape(-1,28,28,1)

train.shape
test = np.array(test)/255

test = test.reshape(-1,28,28,1)

test.shape
from sklearn.model_selection import train_test_split

train_X,valid_X,train_ground,valid_ground = train_test_split(train,

                                                             train,

                                                             test_size=0.2,

                                                             random_state=13)
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
def encoder(input_img):

    #encoder

    #input = 28 x 28 x 1 (wide and thin)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32

    conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64

    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)

    conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    conv4 = BatchNormalization()(conv4)

    return conv4
def fc(enco):

    flat = Flatten()(enco)

    den = Dense(128, activation='relu')(flat)

    out = Dense(num_classes, activation='softmax')(den)

    return out
encode = encoder(input_img)

full_model = Model(input_img,fc(encode))
for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):

    l1.set_weights(l2.get_weights())
autoencoder.get_weights()[0][1]
full_model.get_weights()[0][1]
for layer in full_model.layers[0:19]:

    layer.trainable = False
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
full_model.summary()
train_Y_one_hot = to_categorical(label)

test_Y_one_hot = to_categorical(label)

train_X,valid_X,train_label,valid_label = train_test_split(train,train_Y_one_hot,test_size=0.3,random_state=102)
classify_train = full_model.fit(train_X, train_label, batch_size=256,epochs=20,verbose=2,validation_data=(valid_X, valid_label))
full_model.save_weights('classification_complete.h5')
predicted_classes = full_model.predict(test)
preds = np.argmax(predicted_classes,axis=1)
results = pd.Series(preds,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



print(submission)



submission.to_csv("submission.csv",index=False)