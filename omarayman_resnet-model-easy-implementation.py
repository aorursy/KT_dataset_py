# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge
from keras.optimizers import SGD
from keras.objectives import sparse_categorical_crossentropy as scc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
x_train = (train.ix[:,1:].values).astype('float32') # all pixel values
y_train = train.ix[:,0].values.astype('int32')
test.head()

x_test = test.values.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0
X_train = x_train.reshape(x_train.shape[0],1, 28, 28)
X_test = x_test.reshape(x_test.shape[0],1, 28, 28)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
def get_resnet():
    input_img = Input((1,28,28),name='input_layer')
    zeroPad1 = ZeroPadding2D((1,1), name='zeroPad1', dim_ordering='th')
    zeroPad1_2 = ZeroPadding2D((1,1), name='zeroPad1_2', dim_ordering='th')
    layer1 = Convolution2D(6, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv', dim_ordering='th')
    layer1_2 = Convolution2D(16, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv2', dim_ordering='th')
    zeroPad2 = ZeroPadding2D((1,1), name='zeroPad2', dim_ordering='th')
    zeroPad2_2 = ZeroPadding2D((1,1), name='zeroPad2_2', dim_ordering='th')
    layer2 = Convolution2D(6, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv', dim_ordering='th')
    layer2_2 = Convolution2D(16, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv2', dim_ordering='th')
    zeroPad3 = ZeroPadding2D((1,1), name='zeroPad3', dim_ordering='th')
    zeroPad3_2 = ZeroPadding2D((1,1), name='zeroPad3_2', dim_ordering='th')
    layer3 = Convolution2D(6, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv', dim_ordering='th')
    layer3_2 = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv2', dim_ordering='th')
    layer4 = Dense(64, activation='relu', init='he_uniform', name='dense1')
    layer5 = Dense(16, activation='relu', init='he_uniform', name='dense2')
    final = Dense(10, activation='softmax', init='he_uniform', name='classifier')
######
    first = zeroPad1(input_img)
    second = layer1(first)
    second = BatchNormalization(axis=1, name='major_bn')(second)
    second = Activation('relu', name='major_act')(second)

    third = zeroPad2(second)
    third = layer2(third)
    third = BatchNormalization(axis=1, name='l1_bn')(third)
    third = Activation('relu', name='l1_act')(third)

    third = zeroPad3(third)
    third = layer3(third)
    third = BatchNormalization(axis=1, name='l1_bn2')(third)
    third = Activation('relu', name='l1_act2')(third)


    res = merge([third, second], mode='sum', name='res')


    first2 = zeroPad1_2(res)
    second2 = layer1_2(first2)
    second2 = BatchNormalization(axis=1, name='major_bn2')(second2)
    second2 = Activation('relu', name='major_act2')(second2)


    third2 = zeroPad2_2(second2)
    third2 = layer2_2(third2)
    third2 = BatchNormalization(axis=1, name='l2_bn')(third2)
    third2 = Activation('relu', name='l2_act')(third2)

    third2 = zeroPad3_2(third2)
    third2 = layer3_2(third2)
    third2 = BatchNormalization(axis=1, name='l2_bn2')(third2)
    third2 = Activation('relu', name='l2_act2')(third2)

    res2 = merge([third2, second2], mode='sum', name='res2')

    res2 = Flatten()(res2)

    res2 = layer4(res2)
    res2 = Dropout(0.4, name='dropout1')(res2)
    res2 = layer5(res2)
    res2 = Dropout(0.4, name='dropout2')(res2)
    res2 = final(res2)
    model = Model(input=input_img, output=res2)
    
    
    sgd = SGD(decay=0., lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss=scc, optimizer=sgd, metrics=['accuracy'])
    return model

res = get_resnet()
res.summary()
history = res.fit(X_train, Y_train, validation_split=0.1,verbose=2,epochs=26,batch_size=32)
predicted_labels = res.predict(X_test)


predicted_labels = np.argmax(predicted_labels,axis=1)

predicted_labels=pd.Series(predicted_labels,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predicted_labels],axis = 1)
submission.to_csv("ResNet.csv",index=False)