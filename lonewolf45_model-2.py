



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.models import Sequential,Model

from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,BatchNormalization,Input,Dropout,Activation,LeakyReLU

import keras.backend as K

from keras.layers.merge import add

from keras import regularizers

from sklearn.model_selection import train_test_split as split



from keras.preprocessing.image import ImageDataGenerator as IMGG

from keras.utils import to_categorical

from sklearn.preprocessing import normalize



def gendata(train,test):

    train_x=np.array(train.iloc[:,1:].values)

    train_y=np.array(train.iloc[:,0].values)

    train_y=to_categorical(train_y,num_classes=10)

    test_x=np.array(test.iloc[:,:].values)

    print(test_x.shape)

    train_x=train_x.astype("float64").reshape(-1,28,28,1)

    test_x=test_x.astype("float64").reshape(test_x.shape[0],28,28,1)

    return train_x,train_y,test_x

    



import os

print(os.listdir("../input"))



train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



train_x,train_y,test_x=gendata(train,test)

train_x,val_x,train_y,val_y=split(train_x,train_y,test_size=0.1)

datagen=IMGG(

    rotation_range=15,

    width_shift_range=0.15,

    height_shift_range=0.15,

    rescale=0.3,

    horizontal_flip=False,

    vertical_flip=False

    )

datagen.fit(train_x)



#___________________________________________Model_________________________________________

inp=Input(shape=(28,28,1))

x=inp

x=BatchNormalization()(x)

x=Conv2D(filters=4,strides=1,kernel_size=(1,1),padding="same",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=Conv2D(filters=4,strides=1,kernel_size=(1,1),padding="same",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=MaxPool2D(pool_size=(1,1))(x)

x=BatchNormalization()(x)

x=Conv2D(filters=32,strides=1,kernel_size=(5,5),padding="same",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=Conv2D(filters=32,strides=1,kernel_size=(5,5),padding="same",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=MaxPool2D(pool_size=(2,2))(x)

x=BatchNormalization()(x)

x=Dropout(rate=0.2)(x)

x=LeakyReLU(alpha=0.3)(x)

x=Conv2D(filters=64,strides=1,kernel_size=(5,5),padding="same",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=Conv2D(filters=64,strides=1,kernel_size=(5,5),padding="same",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=MaxPool2D(pool_size=(2,2))(x)

x=BatchNormalization()(x)

x=Dropout(rate=0.25)(x)



x=Flatten()(x)

x=Dense(256,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=LeakyReLU(alpha=0.3)(x)

x=Dropout(rate=0.35)(x)

x=Dense(512,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)

x=LeakyReLU(alpha=0.3)(x)

x=Dropout(rate=0.44)(x)

x=Dense(10,activation="softmax")(x)

resnet=Model(inp,x,name="CNN model")







from keras.callbacks import ReduceLROnPlateau

rlr=ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=4,min_lr=1e-10)

callbacks=[rlr]



resnet.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])



resnet.fit_generator(datagen.flow(train_x,train_y,batch_size=512),epochs=20,validation_data=(val_x,val_y),steps_per_epoch=256,callbacks=callbacks)









#__________________________________________________________________________________________

y_pred=resnet.predict(test_x)



submitable = y_pred.argmax(axis=1)

samsub=pd.read_csv("../input/sample_submission.csv")

samsub["Label"]=submitable

samsub.to_csv("submission.csv",index=False)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.



