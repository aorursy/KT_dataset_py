# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
label = train_data['label']
train_data.drop('label',axis=1,inplace=True)
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Conv2D,AveragePooling2D,GlobalMaxPooling2D,MaxPooling2D,Flatten
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import asarray
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
%matplotlib inline
def identity_block(X,f,filters,stage,block):
    
    
    #defing names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' +  str(stage) + block + '_branch'
    F1,F2,F3 = filters
    
    X_shortcut = X
    
    #first component
    X = Conv2D(filters=F1,kernel_size=(1,1),strides = (1,1),padding='valid',name = conv_name_base + '2a',kernel_initializer= keras.initializers.glorot_uniform())(X)
    X = BatchNormalization(axis = 3,name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    #second component
    X = Conv2D(filters=F2,kernel_size=(f,f),strides = (1,1),padding='same',name = conv_name_base + '2b',kernel_initializer= keras.initializers.glorot_uniform())(X)
    X = BatchNormalization(axis = 3,name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    #thrid component
    X = Conv2D(filters=F3,kernel_size=(1,1),strides = (1,1),padding='valid',name = conv_name_base + '2c',kernel_initializer= keras.initializers.glorot_uniform())(X)
    X = BatchNormalization(axis = 3,name=bn_name_base + '2c')(X)
    
    #Adding shortcut Value to main Path
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    
    return(X)
def convolutional_block(X,f,filters,stage,block,s=2):
    
    #defining names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #filters
    F1,F2,F3 = filters
    
    #store input value
    X_shortcut = X
    
    #first component
    X = Conv2D(filters=F1,kernel_size=(1,1), strides = (s,s), padding= 'valid', name = conv_name_base + '2a',kernel_initializer = keras.initializers.glorot_uniform())(X)
    X = BatchNormalization(axis = 3,name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    #second component
    X = Conv2D(filters=F2,kernel_size=(f,f),strides = (1,1),padding='same',name = conv_name_base + '2b',kernel_initializer= keras.initializers.glorot_uniform())(X)
    X = BatchNormalization(axis = 3,name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    #thrid component
    X = Conv2D(filters=F3,kernel_size=(1,1),strides = (s,s),padding='valid',name = conv_name_base + '2c',kernel_initializer= keras.initializers.glorot_uniform())(X)
    X = BatchNormalization(axis = 3,name=bn_name_base + '2c')(X)
    
    #shortcut path
    X_shortcut = Conv2D(filters=F3,kernel_size=(1,1),strides = (s,s),padding='valid',name = conv_name_base + '1',kernel_initializer= keras.initializers.glorot_uniform())(X_shortcut)
    X = BatchNormalization(axis = 3,name=bn_name_base + '1')(X_shortcut)
    
    #Adding shortcut Value to main Path
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    
    return(X)
    
def ResNet50(input_shape= (28,28,1),classes = 10):
    
    #input as a tensor with shape_input
    X_input = Input(input_shape)
    
    #Zero_Padding 
    X = ZeroPadding2D((3,3))(X_input)
    
    
    #stage 1
    X = Conv2D(64,(3,3),strides = (2,2) , name='conv1' ,kernel_initializer = keras.initializers.glorot_uniform())(X)
    X = BatchNormalization(axis = 3 , name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides = (2,2))(X)
    
    #stage 2
    X = convolutional_block(X , f=3,filters=[64, 64, 256],stage=2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'b')
    X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'c')
    
    #stage 3
    X = convolutional_block(X , f=3, filters =[128,128,512], stage =3, block = 'a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block = 'b')
    X = identity_block(X, 3, [128,128,512], stage=3, block = 'c')
    X = identity_block(X, 3, [128,128,512], stage=3, block = 'd')
    
    #stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage = 4 ,block ='a')
    X = identity_block(X, f=3, filters =[256, 256, 1024],stage = 4, block = 'b')
    X = identity_block(X, f=3, filters =[256, 256, 1024],stage = 4, block = 'c')
    X = identity_block(X, f=3, filters =[256, 256, 1024],stage = 4, block = 'd')
    X = identity_block(X, f=3, filters =[256, 256, 1024],stage = 4, block = 'e')
    X = identity_block(X, f=3, filters =[256, 256, 1024],stage = 4, block = 'f')
    
    #stage 5
    X = convolutional_block(X , f=3, filters = [512, 512, 2048],stage = 5, block = 'a')
    X = identity_block(X, f=3, filters =[512, 512, 2048],stage = 5, block = 'b')
    X = identity_block(X, f=3, filters =[512, 512, 2048],stage = 5, block = 'c')
    

    #output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name = 'fc'+ str(classes), kernel_initializer = keras.initializers.glorot_uniform())(X)
    
    #model
    model = Model(inputs = X_input, outputs = X, name= 'ResNet50')
    
    return model
model = ResNet50(input_shape = (28,28,1), classes = 10)
opt = RMSprop(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#converting flatten data into 28*28 pixel data
train_data = asarray(train_data)
test_data = asarray(test_data)
x_train = train_data.reshape(train_data.shape[0], 28,28, 1) 
x_test = test_data.reshape(test_data.shape[0], 28,28, 1) 

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(label) 
callback = EarlyStopping(monitor='loss',patience=4)
history = model.fit(x_train,y_train,epochs=4,batch_size=32,callbacks=[callback])
#evalution of test dataset on train dataset
model.evaluate(x_train,y_train)
#prediction for test dataset 
prediction = model.predict(x_test)
result = []
for i in prediction:
    result.append(list(i).index(np.max(i)))
ImageId = [i for i in range(1,28001)]
submission = pd.DataFrame({'ImageId':ImageId,'Label':result})
submission.to_csv('submission.csv',index = False)
model.save('my_model.h5')
