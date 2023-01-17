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
import cv2

%matplotlib inline

import matplotlib.pyplot as plt

import keras

import keras.backend as K

from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization,Activation

from keras.layers.merge import add

from keras.models import Model,Sequential

from sklearn.model_selection import train_test_split
CHANNEL_AXIS = 3

stride = 1
#This is a basic Residual Layer





def res_layer(x,temp,filters,pooling = False,dropout = 0.0):

    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)

    temp = BatchNormalization(axis = CHANNEL_AXIS)(temp)

    temp = Activation("relu")(temp)

    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)



    x = add([temp,Conv2D(filters,(3,3),strides = stride,padding = "same")(x)])

    if pooling:

        x = MaxPooling2D((2,2))(x)

    if dropout != 0.0:

        x = Dropout(dropout)(x)

    temp = BatchNormalization(axis = CHANNEL_AXIS)(x)

    temp = Activation("relu")(temp)

    return x,temp
inp = Input(shape = (32,32,3))

x = inp

x = Conv2D(16,(3,3),strides = stride,padding = "same")(x)

x = BatchNormalization(axis = CHANNEL_AXIS)(x)

x = Activation("relu")(x)

temp = x

#from here on stack the residual layers remember to use padding only while increasing no. of filters.

x,temp = res_layer(x,temp,32,dropout = 0.2)

x,temp = res_layer(x,temp,32,dropout = 0.3)

x,temp = res_layer(x,temp,32,dropout = 0.4,pooling = True)

x,temp = res_layer(x,temp,64,dropout = 0.2)

x,temp = res_layer(x,temp,64,dropout = 0.2,pooling = True)

x,temp = res_layer(x,temp,256,dropout = 0.4)

x = temp

x = Flatten()(x)

x = Dropout(0.4)(x)

x = Dense(256,activation = "relu")(x)

x = Dropout(0.23)(x)

x = Dense(128,activation = "relu")(x)

x = Dropout(0.3)(x)

x = Dense(64,activation = "relu")(x)

x = Dropout(0.2)(x)

x = Dense(32,activation = "relu")(x)

x = Dropout(0.2)(x)

x = Dense(14,activation = "softmax")(x)



resnet_model = Model(inp,x,name = "Resnet")

resnet_model.summary()





X_train = np.load("../input/wildcam-reduced/X_train.npy")

Y_train = np.load("../input/wildcam-reduced/y_train.npy")

X_test = np.load("../input/wildcam-reduced/X_test.npy")
X_train = X_train.astype("float32")/255.0

X_test = X_test.astype("float32")/255.0
from keras.callbacks import ReduceLROnPlateau



reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,patience=5, min_lr=1e-5)

callbacks = [reduce_lr]

resnet_model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
resnet_model.fit(X_train,Y_train,batch_size=200,epochs = 15,validation_split=0.18,callbacks = callbacks)
resnet_model.fit(X_train,Y_train,batch_size=350,epochs = 10,validation_split=0.5,callbacks = callbacks)
os.listdir("../input")
pred = resnet_model.predict(X_test)
submission_df = pd.read_csv("../input/iwildcam-2019-fgvc6/sample_submission.csv")

submission_df.head()
train_df = pd.read_csv("../input/iwildcam-2019-fgvc6/train.csv")

train_df.describe()
d = {i:0 for i in range(23)}
for c in train_df["category_id"]:

    d[c]+=1

d
matched_dict = {i:0 for i in range(14)}

k = 0

for i in range(23):

    if d[i] > 0:

        matched_dict[k] = i

        k+=1

matched_dict
submittable = pred.argmax(axis = 1)

print(submittable.shape)
for i in range(submittable.shape[0]):

    submittable[i] = matched_dict[submittable[i]]
submission_df["Predicted"] = submittable

submission_df.head()
submission_df.describe()
est = {i:0 for i in range(23)}

for i in submittable:

    est[i]+=1

est
resnet_model.save("resnet.h5")
resnet_model.save_weights("resnet_weights.h5")
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

datagen.fit(X_train)

X_batch = datagen.flow(X_train, Y_train, batch_size=1000)
history = resnet_model.history

history.history.keys()
plt.plot(history.epoch,history.history["val_acc"])
submission_df.to_csv('submission_resnet2.csv',index=False)
