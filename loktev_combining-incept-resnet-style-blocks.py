# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Activation, concatenate, add, Dense, Dropout, Conv2D, BatchNormalization, Flatten, MaxPooling2D

from tensorflow.keras.layers import PReLU

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.python.keras.optimizers import Adam

from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dataset = np.loadtxt('../input/train.csv', skiprows=1, delimiter=',')
x_train = train_dataset[:, 1:]

len(x_train)
x_train=x_train.reshape((42000,28,28,1))
y_train = train_dataset[:, 0]

y_train = utils.to_categorical(y_train)
class NetMnist:

  

    def fit_gen(self,train,datagen):

            self.model.compile(metrics=['accuracy'],loss='categorical_crossentropy',optimizer=Adam(lr=2e-3))

            learning_rate_reduction = ReduceLROnPlateau(monitor = 'acc', 

                                            patience = 3, 

                                            verbose = 1, 

                                            factor = 0.25, 

                                            min_lr = 0.00001)

            self.model.fit_generator(datagen.flow(train[0],train[1], batch_size=200),

                    steps_per_epoch=len(train[0]) //200, 

                    callbacks = [learning_rate_reduction],

                    epochs=60)

            self.model.compile(metrics=['accuracy'],loss='categorical_crossentropy',optimizer=Adam(lr=1e-5))

            self.model.fit_generator(datagen.flow(train[0],train[1], batch_size=200),

                    steps_per_epoch=len(train[0]) //200, 

                    epochs=5)

       

    

    def predvec(self,data):

        return self.model.predict(data)  

  





#incept-style block

def incept_block(inp,d):

  a = Conv2D(1, (1, 1), padding='same')(inp)

  x = Conv2D(d, (3, 3), padding='same')(inp)

  y = Dropout(0.1)(x)

  y = BatchNormalization()(y)

  y = PReLU()(y)

  y = Conv2D(d, (3, 3), padding='same')(y)

  r = concatenate([a,x,y])

  r = Dropout(0.1)(r)

  r = BatchNormalization()(r)

  return(PReLU()(r))
#resnet-style block

def resnet_block(inp,d):

  x = Conv2D(d, (3, 3), padding='same')(inp)

  x = Dropout(0.1)(x)

  x = BatchNormalization()(x)

  y = PReLU()(x)

  y = Conv2D(d, (3, 3), padding='same')(y)

  y = Dropout(0.1)(y)

  y = BatchNormalization()(y)

  a = Conv2D(1, (1, 1), padding='same')(inp)

  r = add([y,a])

  return(PReLU()(r))
class CNN(NetMnist):

  

  def __init__(self,block1,block2,c,d,dd):

    input_tensor = Input((28, 28,1))

    x = BatchNormalization(input_shape=(28, 28, 1))(input_tensor)

    x = block1(x,c,)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = block2(x,2*c)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(d)(x)

    x = Dropout(dd)(x)

    x = PReLU()(x)

    x=Dense(10, activation='softmax')(x)

    self.model=Model(inputs=input_tensor, outputs=x)

    

    

    print('Network with {} layers'.format(len(self.model.layers)))
m=[CNN(incept_block,incept_block,32,512,0.2),CNN(incept_block,incept_block,32,512,0.2),CNN(resnet_block,incept_block,32,512,0.2),CNN(resnet_block,incept_block,32,512,0.2),CNN(incept_block,resnet_block,32,512,0.2),CNN(incept_block,resnet_block,32,512,0.2), CNN(resnet_block,resnet_block,32,512,0.2),CNN(resnet_block,resnet_block,32,512,0.2) ] 
datagen = [ImageDataGenerator(

    rotation_range=12,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.1),

    

           ImageDataGenerator(

    rotation_range=8,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2)]

    



datagen[0].fit(x_train)

datagen[1].fit(x_train)
for i in range(len(m)):

    m[i].fit_gen((x_train, y_train),datagen[i%2])
test_dataset = np.loadtxt('../input/test.csv', skiprows=1, delimiter=",")
#x_test = test_dataset / 255.0
x_test = test_dataset.reshape(test_dataset.shape[0], 28, 28, 1)
v=m[0].predvec(x_test)

for i in range(1,len(m)):

    v+=m[i].predvec(x_test)
predictions=np.argmax(v,axis=1)
out = np.column_stack((range(1, predictions.shape[0]+1), predictions))

np.savetxt('submission.csv', out, header="ImageId,Label", 

            comments="", fmt="%d,%d")
!head submission.csv