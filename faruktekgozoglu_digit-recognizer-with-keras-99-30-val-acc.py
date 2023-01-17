# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras as K
from keras.utils import to_categorical 
from keras.layers import Dense,Flatten,Conv2D,Dropout,BatchNormalization,Activation,Input,MaxPooling2D
from keras import regularizers
from keras.optimizers import Adam
from keras.models import Model

import matplotlib.pyplot as plt
print(tf.test.gpu_device_name())
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
Y_train = train_data['label']

X_train = train_data.drop(labels = ['label'],axis=1).values
X_train.shape
X_train = X_train.reshape((X_train.shape[0],28,28))
X_train = np.expand_dims(X_train, axis=3)
X_train.shape
nb_class = 10
Y_train.shape
Y_train = to_categorical(Y_train,num_classes = nb_class)
Y_train.shape
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=42)
print("Shape of train set: " + str(X_train.shape))
print("Shape of validation set: " + str(X_val.shape))
print("Shape of train set labels: " + str(Y_train.shape))
print("Shape of validation set labels: " + str(Y_val.shape))
del train_data
X_train = X_train.astype(float)
X_val = X_val.astype(float)
X_train /= 255
X_val /= 255
def conv_block(input_tensor,
               filters,
               kernel_size,
               stage=None,
               block=None,
               padding='same',
               strides=(1,1),
               batch_mom=0.0,
               batch_axis=3,
               initializer='RandomNormal'):
    
    conv_name_base = 'model_' + str(stage) + '_' + str(block)
    
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding=padding,
               strides=strides,
               kernel_initializer=initializer,
               name = conv_name_base)(input_tensor)
    x = BatchNormalization(axis=batch_axis,momentum=batch_mom)(x)
    x = Activation('relu')(x)
    return x
    
input_shape = (28,28,1) 
initializer='glorot_normal'
reg_val=0.0

main_input = Input(shape=(input_shape),name='input_of_network')
x = conv_block(main_input,
               filters=32,
               kernel_size=(3,3),
              stage='s1',
              block='3_3',
              padding='valid')

y = conv_block(main_input,
              filters=32,
              kernel_size=(3,3),
              stage='s2',
              block='3_3',
              padding='valid',
              strides=(2,2))

z = conv_block(main_input,
              filters=96,
              kernel_size=(3,3),
              stage='s3',
              block='3_3',
              padding='valid',
              strides=(3,3))

x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

x_y = K.layers.concatenate([x,y])

x_y = conv_block(x_y,
              filters=96,
              kernel_size=(1,1),
              stage='2',
              block='1_1',
              padding='same',
              strides=(1,1))

x_y = conv_block(x_y,
              filters=128,
              kernel_size=(3,3),
              stage='2',
              block='3_3',
              padding='valid',
              strides=(1,1))

x_y = conv_block(x_y,
              filters=144,
              kernel_size=(3,3),
              stage='3',
              block='3_3',
              padding='valid',
              strides=(1,1))

x_y_z = K.layers.concatenate([x_y,z])

out = Flatten()(x_y_z)
out = Dense(256,kernel_initializer=initializer,kernel_regularizer=regularizers.l2(reg_val))(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Dropout(0.3)(out)

out = Dense(256,kernel_initializer=initializer,kernel_regularizer=regularizers.l2(reg_val))(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Dropout(0.3)(out)

out = Dense(10,activation='softmax')(out)

model = Model(main_input,out)

model.summary()

adam = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer=adam)
model.fit(x=X_train,
         y=Y_train,
         shuffle=True,
        batch_size = 256,
         validation_data=(X_val,Y_val),
         epochs=100,
         verbose=1)
plt.plot(model.history.history['acc'],'r-')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()
plt.plot(model.history.history['loss'],'b-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.plot(model.history.history['val_acc'],'r-')
plt.xlabel('Epochs')
plt.ylabel('Validation Acc.')
plt.show()
plt.plot(model.history.history['val_loss'],'b-')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.show()
test_data.shape
test_data = test_data.values
test_data = np.reshape(test_data, (test_data.shape[0],28,28)).astype(float)
test_data /= 255
test_data = np.expand_dims(test_data, axis=3)
test_data.shape
y_pred = model.predict(test_data)
y_pred_submission = np.argmax(y_pred,axis = 1)
y_pred_submission = y_pred_submission.astype(int)
y_pred_submission[:10]
np.savetxt('submission.csv',y_pred_submission,delimiter=',')