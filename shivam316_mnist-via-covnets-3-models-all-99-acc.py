# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
X=train.drop('label',axis=1)/255.0
testX=test/255.0
y=train.label
np.random.seed(42)
tf.random.set_seed(42)
k=keras.backend
k.clear_session()
from sklearn.model_selection import train_test_split
Xtrain,Xval,ytrain,yval=train_test_split(X,y,test_size=0.3,random_state=42)
Xtrain,Xtest,ytrain,ytest=train_test_split(Xtrain,ytrain,test_size=0.1,random_state=42)
Xtrain,Xval,Xtest=tf.Variable(Xtrain),tf.Variable(Xval),tf.Variable(Xtest)
Xtrainpp=tf.reshape(Xtrain,[Xtrain.shape[0],28,28,1])
Xvalpp=tf.reshape(Xval,[Xval.shape[0],28,28,1])
Xtestpp=tf.reshape(Xtest,[Xtest.shape[0],28,28,1])
testX=tf.reshape(testX,[testX.shape[0],28,28,1])
my_callbacks = [
    keras.callbacks.EarlyStopping(patience=5),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.0001),
]
from functools import partial

Default2D=partial(keras.layers.Conv2D,kernel_size=3,activation='relu',padding='SAME')
MaxPool2D=partial(keras.layers.MaxPool2D,pool_size=2)
model_vgg=keras.models.Sequential([
    Default2D(filters=32,kernel_size=5,input_shape=[28,28,1]),
    MaxPool2D(),
    Default2D(filters=64),
    MaxPool2D(),
    Default2D(filters=128),
    MaxPool2D(),
    Default2D(filters=256,kernel_size=2),
    MaxPool2D(),
    
    keras.layers.Flatten(),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(25,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10,activation='softmax')
])
model_vgg.summary()
model_vgg.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='nadam')
history_vgg=model_vgg.fit(Xtrainpp,ytrain,epochs=20,validation_data=(Xvalpp,yval),callbacks=my_callbacks)
model_vgg.evaluate(Xtestpp,ytest)
model_vgg.save('mnist_cnn_vgg_992_007.h5') #99.2% accu and 0.07 % loss
class Inseption2D(keras.layers.Layer):
    def __init__(self,f11,f311,f333,f511,f555,fMP11,**kwargs):
        super().__init__(**kwargs)
        self.f11=f11
        self.f311=f311
        self.f333=f333
        self.f511=f511
        self.f555=f555
        self.fMP11=fMP11
        
        self.Conv1x1=keras.layers.Conv2D(filters=self.f11,kernel_size=1,activation='relu',padding='same')
    
        self.Conv3SL1x1=keras.layers.Conv2D(filters=self.f311,kernel_size=1,activation='relu',padding='same')
        self.Conv3SL3x3=keras.layers.Conv2D(filters=self.f333,kernel_size=3,activation='relu',padding='same')
        
        self.Conv5SL1x1=keras.layers.Conv2D(filters=self.f511,kernel_size=1,activation='relu',padding='same')
        self.Conv5SL5x5=keras.layers.Conv2D(filters=self.f555,kernel_size=5,activation='relu',padding='same')
        
        self.MaxPool=keras.layers.MaxPooling2D(pool_size=3,strides=1,padding='same')
        self.ConvMP1x1=keras.layers.Conv2D(filters=self.fMP11,kernel_size=1,activation='relu',padding='same')
        
    def call(self,inputs):
        #Input via 1x1
        out11=self.Conv1x1(inputs)
        
        #Input via Smart Layer (1x1,3x3)
        x=self.Conv1x1(inputs)
        out33=self.Conv3SL1x1(x)
        
        #Input via Smart Layer (1x1,5x5)
        x=self.Conv1x1(inputs)
        out55=self.Conv5SL1x1(x)
        
        #Input via Max Pool
        x=self.MaxPool(inputs)
        outMP11=self.ConvMP1x1(x)
        
        #concat the outputs
        output=keras.layers.Concatenate(axis=-1)([out11,out33,out55,outMP11])
        
        return output
    def get_config(self):
        base_config=super().get_config()
        return {**base_config,
                'f11':self.f11,'f311':self.f311,'f333':self.f333,
                'f511':self.f511,'f555':self.f555,'fMP11':self.fMP11}
from functools import partial
Default2D=partial(keras.layers.Conv2D,kernel_size=3,activation='relu',padding='same')
MaxPool2D=partial(keras.layers.MaxPool2D,pool_size=2,padding='same')
model_gnet=keras.models.Sequential([
    Default2D(filters=64,kernel_size=7,input_shape=[28,28,1]),
    MaxPool2D(),
    Default2D(filters=32,kernel_size=1),
    Default2D(filters=128),
    MaxPool2D(),
    Inseption2D(f11=32,f311=16,f333=64,f511=16,f555=32,fMP11=16),
    Inseption2D(f11=64,f311=32,f333=96,f511=32,f555=64,fMP11=32),
    MaxPool2D(),
    Inseption2D(f11=96,f311=64,f333=108,f511=64,f555=96,fMP11=64),
    Inseption2D(f11=108,f311=96,f333=128,f511=64,f555=108,fMP11=64),
    MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(90,activation='relu'),
    keras.layers.Dense(45,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model_gnet.summary()
model_gnet.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='nadam')
history_gnet=model_gnet.fit(Xtrainpp,ytrain,epochs=20,validation_data=(Xvalpp,yval),callbacks=my_callbacks)
model_gnet.evaluate(Xtestpp,ytest)
model_gnet.save('mnist_cnn_inseption_988_005.h5') #98.84%  0.05%
class ResidualBlock(keras.layers.Layer):
    def __init__(self,filters,strides=1,activation='relu',**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.strides=strides
        self.activation=keras.activations.get(activation)
        self.main_layers=[
            keras.layers.Conv2D(filters,2,strides=strides,padding='same',use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters,2,strides=1,padding='same',use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers=[]
        if strides>1:
            self.skip_layers=[
                keras.layers.Conv2D(filters,1,strides=strides,padding='same',use_bias=False),
                keras.layers.BatchNormalization()
            ]
    def call(self,inputs):
        Z=inputs
        for layer in self.main_layers:
            Z=layer(Z)
        skip_Z=inputs
        for layer in self.skip_layers:
            skip_Z=layer(skip_Z)
        return self.activation(Z+skip_Z)
    
    def get_config(self):
        base_config=super().get_config()
        return {**base_config,"filters":self.filters,"strides":self.strides,"activation":keras.activations.serialize(self.activation)}
from functools import partial
Default2D=partial(keras.layers.Conv2D,kernel_size=3,activation='relu',padding='same')
MaxPool2D=partial(keras.layers.MaxPool2D,pool_size=2,padding='same')
model_rnet=keras.models.Sequential([
    Default2D(filters=32,kernel_size=5,input_shape=[28,28,1]),
    MaxPool2D(),
    ResidualBlock(filters=64,strides=2),
    ResidualBlock(filters=64),
    ResidualBlock(filters=128,strides=2),
    ResidualBlock(filters=128),
    MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(25,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model_rnet.summary()
model_rnet.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='nadam')
history_rnet=model_rnet.fit(Xtrainpp,ytrain,epochs=20,validation_data=(Xvalpp,yval),callbacks=my_callbacks)
model_rnet.evaluate(Xtestpp,ytest)
model_rnet.save('mnist_cnn_residual_99_003.h5') #99% 0.03%
ypred=your_model_name.predict_classes(your_prepared_test_data)
ImageId=pd.Series(range(1,28001))
Label=pd.Series(ypred)
sol=pd.concat([ImageId, Label],axis=1)
sol=sol.rename(columns={0: "ImageId", 1: "Label"})
sol.to_csv('mnist_via_vggnet_sol.csv',index=False)
"""

class DepthMaxPool(keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
    def call(self, inputs):
        return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)

"""
"""

from functools import partial

Default2D=partial(keras.layers.Conv2D,kernel_size=3,activation='relu',padding='SAME')
MaxPool2D=partial(keras.layers.MaxPooling2D,pool_size=2)
model=keras.models.Sequential([
    Default2D(filters=90,kernel_size=7,input_shape=[28,28,1]),
    MaxPool2D(),
    Default2D(filters=180),
    MaxPool2D(),
    Default2D(filters=256),
    DepthMaxPool(16),
    Default2D(filters=360),
    MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(90,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(45,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10,activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='nadam')
history=model.fit(Xtrainpp,ytrain,epochs=5,validation_data=(Xvalpp,yval))
score=model.evaluate(Xtestpp,ytest)

"""