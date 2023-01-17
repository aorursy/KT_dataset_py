import numpy as np
import pandas as pd
import keras
from keras.callbacks import ReduceLROnPlateau
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras import regularizers
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import os
def inputs(train_path='../input/train.csv', test_path='../input/test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    valid_X = test.values
    valid_X = valid_X / 255
    t_X = train.drop("label", axis=1)
    t_Y = train["label"]
    t_X = np.reshape(t_X.values, (-1,28,28,1))
    valid_X = np.reshape(valid_X, (-1,28,28,1))
    train_X, test_X, train_Y, test_Y = train_test_split(t_X, t_Y, test_size=0.2, random_state=0)
    train_X = train_X / 255
    test_X = test_X / 255    
    train_Y = tf.one_hot(train_Y, 10)
    test_Y = tf.one_hot(test_Y, 10)   
    sess = tf.Session()
    train_Y, test_Y = sess.run([train_Y, test_Y])
    sess.close()
    return train_X, test_X, train_Y, test_Y, valid_X

def model(input_shape):
    X_input = Input(input_shape)
    #X = ZeroPadding2D((3, 3))(X_input)
    # CONV -> CONV -> BN -> MAXPOOL -> DropOut
    X = Conv2D(32, (7, 7), strides = (1, 1),padding = 'Same',name = 'conv0')(X_input) 
    X = BatchNormalization()(X) 
    X = Activation('relu')(X)
    X = Conv2D(64, (4, 4), strides = (1, 1),padding = 'Same', name = 'conv1')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    X = Dropout(0.4)(X)
    
    # CONV -> CONV -> BN -> MAXPOOL -> DropOut
    X = Conv2D(128, (4, 4), strides = (1,1),padding = 'Same', name = 'conv2')(X) 
    X = BatchNormalization()(X) 
    X = Activation('relu')(X)
    X = Conv2D(256, (2, 2), strides = (2, 2),padding = 'Valid', name = 'conv3')(X)
    X = BatchNormalization()(X) 
    X = Activation('relu')(X)
        
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    X = Dropout(0.4)(X)
    
    # FLATTEN -> Dense -> BN -> DropOut
    X = Flatten()(X)
    X = Dense(257,  name='fc1')(X)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    #Dense -> BN -> DropOut
    X = Dense(126, name='fc2')(X)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.6)(X)
    #Dense 
    X = Dense(10, activation='softmax', name='fcf')(X)
    #Model Creation
    model = Model(inputs = X_input, outputs = X, name='Model')

    return model
train_X, test_X, train_Y, test_Y, valid_X = inputs()
datagen = ImageDataGenerator(       
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1)        
datagen.fit(train_X)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,factor=0.5, min_lr=0.00001)
model = model(train_X.shape[1:])
model.summary()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'] )
history = model.fit_generator(datagen.flow(train_X, train_Y, batch_size=32), validation_data=(test_X, test_Y), steps_per_epoch=len(train_X)//32, epochs=45, callbacks=[learning_rate_reduction])
result = model.predict(valid_X)
result = np.argmax(result, 1)
predictions = result.T
result = pd.DataFrame({'ImageId': range(1,len(predictions)+1), 'Label': predictions})
result.to_csv('result.csv', index=False, encoding='utf-8')