import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from matplotlib import pyplot as plt

import cv2

import os

from keras import Sequential

from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout,Activation,ELU
data = []

counter = 0

for path in os.listdir("../input/kkanji/kkanji2"):

    for file in os.listdir(os.path.join("../input/kkanji/kkanji2",path)):

        image_data=cv2.imread(os.path.join("../input/kkanji/kkanji2",path,file), cv2.IMREAD_GRAYSCALE)

        image_data=cv2.resize(image_data,(128,128))

        data.append(image_data)

        counter+=1

        if counter%10000==0:

            print (counter,"image data retrieved")
n = 128

mask = np.array([[(x<=47)|(x>=80)|(y<=47)|(y>=80) for x in range(n)] for y in range(n)])

maskedData = []

for i in range(len(data)):

    maskedImage = np.multiply(mask, data[i])

    maskedData.append(maskedImage)
data=np.array(data)

print (data.shape)

data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],1)

maskedData=np.array(maskedData)

print (maskedData.shape)

maskedData=maskedData.reshape((maskedData.shape)[0],(maskedData.shape)[1],(maskedData.shape)[2],1)
plt.imshow(data[10005,:,:,0])
plt.imshow(np.multiply(mask, data[10005,:,:,0]))
def loss_function(y_true, y_pred):

    return tf.norm(tf.subtract(y_true, y_pred), 1)
import tensorflow as tf

from keras.backend import clip

from keras.layers import Lambda

def UpSampling2DBilinear(size):

    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = 5, strides= 1, dilation_rate= 1, input_shape=(128,128,1), padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 64, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 64, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 2, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 2, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 4, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 8, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 16, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 128, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(UpSampling2DBilinear((128,128)))

model.add(Conv2D(filters = 64, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 64, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 32, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 16, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

model.add(Conv2D(filters = 1, kernel_size = 3, strides= 1, dilation_rate= 1, padding = 'same'))

model.add(Activation(ELU(0.0002)))

#model.add(clip(-1,1))

model.summary()

model.compile(optimizer="adadelta",loss= loss_function)
model.fit(maskedData, data,validation_split=0.2,epochs=3,batch_size=32)