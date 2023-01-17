%matplotlib inline
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import os 
 
from sklearn.metrics import accuracy_score 
import tensorflow as tf 
import keras
df = pd.read_csv('../input/digit-recognizer/train.csv')
df.head()
df1 = df.copy()
df1.drop('label', axis=1,inplace=True)
labels = df['label'].values
img_array = df1.values
img_array = img_array/255
img_array = img_array.reshape(-1,28,28,1)
img_array.shape
seed = 128 
rng = np.random.RandomState(seed)
labels = keras.utils.to_categorical(labels)
split_size = int(img_array.shape[0]*0.7) 
train_x, val_x = img_array[:split_size], img_array[split_size:] 
train_y, val_y = labels[:split_size], labels[split_size:]
from tensorflow.keras.layers import Input, Conv2D,ReLU, Flatten, Dense, MaxPool2D, concatenate, BatchNormalization, AveragePooling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
def inception(layer_in, num_filter):
    conv = Conv2D(num_filter,(3,3),padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    conv2 = Conv2D(num_filter,(5,5),padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    
    max_pool = MaxPool2D(pool_size=(2,2), strides=(1,1), padding='same')(layer_in)
    
    output = concatenate(inputs=[conv, conv2, max_pool], axis=-1)
    output = Activation('relu')(output)
    return output
def residual_net(layer_in, n_filters, final_filter):
    conv2 = inception(layer_in, n_filters)
        
    output = Conv2D(filters=final_filter,kernel_size =1, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    layer_out = Add()([conv2, output])
    
    layer_out = Activation('relu')(layer_out)
    return layer_out
def create_inception():
    inputs = Input(shape=(28,28,1))
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3, strides=1, filters=num_filters, padding="same")(t)
    t = Activation('relu')(t)
    
    #t = residual_net(t,64,192)
    t= inception(t,64)
    #t = residual_net(t, 64,448)
    t = inception(t, 64)
  
    
    t = Conv2D(kernel_size=3, strides=1, filters=num_filters, padding="same")(t)
    t = Activation('relu')(t)
    t = inception(t,32)
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    
    output = Dense(128, activation = 'relu', kernel_initializer = 'RandomUniform')(t)
    output = Dense(64, activation = 'relu', kernel_initializer = 'RandomUniform')(output)
    
    outputs = Dense(10, activation='softmax')(output)
    
    model = Model(inputs, outputs)
    opti = SGD(lr=0.001, momentum=0.9, decay=0.001/50, nesterov=True)
    model.compile(optimizer = opti, loss='categorical_crossentropy', metrics = ['accuracy'])
    
    return model
model = create_inception()
model.summary()
model.fit(x = img_array, y = labels, validation_data=(val_x, val_y),epochs=50,batch_size=128)
test = pd.read_csv('../input/digit-recognizer/test.csv')
test = test.values
test = test.reshape(-1,28,28,1)
test = test/255
pred=model.predict(test)
label = np.argmax(pred, axis=1)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission['Label'] = label
submission.to_csv('submission.csv', index=False)
submission
