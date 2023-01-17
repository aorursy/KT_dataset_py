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
import h5py
import keras
from keras.layers import Conv2D,Dense,Add,Input,Flatten,MaxPool2D,BatchNormalization,Activation,Input,ZeroPadding2D,AveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from matplotlib.pyplot import imshow
def id_block(X,filters,f):

    X_initial = X

    F1,F2,F3 = filters

    

    X = Conv2D(F1,kernel_size=(1,1),strides=(1,1),padding='valid')(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F2,kernel_size=(f,f),strides=(1,1),padding = 'same')(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    

    X = Conv2D(F3,kernel_size=(1,1),strides=(1,1),padding = 'valid')(X)

    X = BatchNormalization(axis = 3)(X)

    

    X = Add()([X,X_initial])

    X = Activation('relu')(X)

    

    return X
def conv(X,filters,f,s = 2):

    F1,F2,F3 = filters

    X_initial = X

    

    X = Conv2D(filters = F1,kernel_size=(1,1),strides=(s,s),padding = 'valid')(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F2,kernel_size=(f,f),strides=(1,1),padding = 'same')(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F3,kernel_size=(1,1),strides=(1,1),padding = 'valid')(X)

    X = BatchNormalization(axis = 3)(X)

    

    X_initial = Conv2D(F3,kernel_size=(1,1),strides=(s,s),padding = 'valid')(X_initial)

    X_initial = BatchNormalization(axis = 3)(X_initial)

    

    X = Add()([X,X_initial])

    X = Activation('relu')(X)

    

    return X
def ResNet50(input_shape = (64,64,3)):

    X_input = Input(shape = input_shape)

    

    X = ZeroPadding2D((3,3))(X_input)

    

    X = Conv2D(filters = 64,kernel_size=(7,7),strides=(2,2))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    X = MaxPool2D(pool_size=(3,3),strides=(2,2))(X)

    

    X = conv(X,[64,64,256],3,1)

    X = id_block(X,[64,64,256],3)

    X = id_block(X,[64,64,256],3)

    

    X = conv(X,[128,128,512],3,2)

    X = id_block(X,[128,128,512],3)

    X = id_block(X,[128,128,512],3)

    X = id_block(X,[128,128,512],3)

    

    X = conv(X,[256,256,1024],3,2)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    

    X = conv(X,[512,512,2048],3,2)

    X = id_block(X,[512,512,2048],3)

    X = id_block(X,[512,512,2048],3)

    

    X = AveragePooling2D()(X)

    X = Flatten()(X)

    X = Dense(6,activation = 'softmax')(X)

    

    model = Model(X_input,X,name = 'ResNet50')

    

    return model
model = ResNet50()
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
train_x,train_y,test_x,test_y,classes = load_data()
train_x = train_x/255

test_x = test_x/255
train_y = to_categorical(train_y).reshape(-1,6)

test_y = to_categorical(test_y).reshape(-1,6)
model.fit(train_x,train_y,epochs = 20,batch_size=128)
acc_stats = model.evaluate(test_x,test_y)
print('Model Accuracy:\nLoss: ',acc_stats[0],'\nAccuracy: ',acc_stats[1]*100,'%')
model.save_weights('rn50.h5')
json_data = model.to_json()
with open('rn50.json','w') as json_file:

    json_file.write(json_data)
def load_data():

    train = h5py.File('../input/train_signs.h5')

    test = h5py.File('../input/test_signs.h5')

    

    train_x = np.array(train['train_set_x'][:])

    train_y = np.array(train['train_set_y'][:])

    test_x = np.array(test['test_set_x'][:])

    test_y = np.array(test['test_set_y'][:])

    classes = np.array(test['list_classes'][:])

    

    train_y = train_y.reshape((1,train_y.shape[0]))

    test_y = test_y.reshape((1,test_y.shape[0]))



    return train_x,train_y,test_x,test_y,classes
def ResNet50_2(input_shape = (64,64,3)):

    X_input = Input(shape = input_shape)

    

    X = ZeroPadding2D((3,3))(X_input)

    

    X = Conv2D(filters = 64,kernel_size=(7,7),strides=(2,2))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    X = MaxPool2D(pool_size=(3,3),strides=(2,2))(X)

    

    X = conv(X,[64,64,256],3,1)

    X = id_block(X,[64,64,256],3)

    X = id_block(X,[64,64,256],3)

    

    X = conv(X,[128,128,512],3,2)

    X = id_block(X,[128,128,512],3)

    X = id_block(X,[128,128,512],3)

    X = id_block(X,[128,128,512],3)

    

    X = conv(X,[256,256,1024],3,2)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    X = id_block(X,[256,256,1024],3)

    

    X = conv(X,[512,512,2048],3,2)

    X = id_block(X,[512,512,2048],3)

    X = id_block(X,[512,512,2048],3)

    

    X = MaxPool2D()(X)

    X = Flatten()(X)

    X = keras.layers.Dropout(0.5)(X)

    X = Dense(6,activation = 'softmax')(X)

    

    model = Model(X_input,X,name = 'ResNet50')

    

    return model
try2 = ResNet50_2()
datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,zoom_range=0.5,rotation_range=0.5)
trainset = datagen.flow(train_x,train_y,shuffle=False)
testset = datagen.flow(test_x,test_y,shuffle = False)
try2.fit_generator(trainset,epochs = 30,steps_per_epoch=50)
try2.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
try2.fit(train_x,train_y,epochs=25,batch_size=64,validation_split=0.25)
try2.evaluate(test_x,test_y,batch_size=1)
try2.save_weights('try5.h5')
try3 = ResNet50_2()
try3.load_weights('../working/try2.h5')
try3.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
try3.evaluate(test_x,test_y)
json_data = try2.to_json()
with open('try5.json','w') as json_file:

    json_file.write(json_data)
json_file.close()