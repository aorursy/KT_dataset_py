# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import BatchNormalization

from keras import backend as K



## Load data into dataframes



import pandas as pd

train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
test.head()
## split train data into train and cv by stratified sampling



from sklearn.model_selection import train_test_split



Y = train['label']

xtrain, xcv, ytrain, ycv = train_test_split(train, Y, test_size=0.33, stratify=Y)
xtrain=xtrain[xtrain.columns[1:785]]

xcv=xcv[xcv.columns[1:785]]
print(xtrain.shape, ytrain.shape)

print(xcv.shape, ycv.shape)
xtrainn=xtrain.to_numpy()

xcvv=xcv.to_numpy()

ytrainn=ytrain.to_numpy()

ycvv=ycv.to_numpy()
xtestt=test.to_numpy()
## Convert class labels to vectors

from keras.utils import np_utils

ytrainn=np_utils.to_categorical(ytrainn, 10)

ycvv=np_utils.to_categorical(ycvv, 10)
print(ytrainn[3])
batchsize=128

numclasses=10

epochs=12



## input image dimensions

imgrows, imgcols=28,28
if K.image_data_format=='channels_first':

    xtrainn=xtrainn.reshape(xtrainn.shape[0], 1, imgrows, imgcols)

    xcvv=xcvv.reshape(xcvv.shape[0], 1, imgrows, imgcols)

    inputshape=(1, imgrows, imgcols)

else:

    xtrainn=xtrainn.reshape(xtrainn.shape[0], imgrows, imgcols, 1)

    xcvv=xcvv.reshape(xcvv.shape[0], imgrows, imgcols, 1)

    inputshape=(imgrows, imgcols, 1)
## Normalize data



xtrainn=xtrainn/255

xcvv=xcvv/255
print(xtrainn.shape, ytrainn.shape)

print(xcvv.shape, ycvv.shape)
## kernel size 3X3, Maxpooling, dropouts 0.3, 0.5



model1=Sequential()

model1.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=inputshape))

model1.add(Conv2D(64, (3,3), activation='relu'))

model1.add(Conv2D(128, (3,3), activation='relu'))

model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Dropout(0.25))



model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dropout(0.25))



model1.add(Dense(numclasses, activation='softmax'))



model1.compile(loss=keras.losses.categorical_crossentropy, 

              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])



model1.fit(xtrainn, ytrainn, batch_size=batchsize, epochs=epochs, verbose=1, 

          validation_data=(xcvv, ycvv))



score1=model1.evaluate(xcvv, ycvv, verbose=0)

print("Validation Loss= ", score1[0])

print("Validation Accuracy= ",score1[1])


## Test class label predictions

if K.image_data_format=='channels_first':

    xtestt=xtestt.reshape(xtestt.shape[0], 1, imgrows, imgcols)

    

else:

    xtestt=xtestt.reshape(xtestt.shape[0], imgrows, imgcols, 1)

print(xtestt.shape)

ytest=model1.predict_classes(xtestt)
ids=np.arange(1,28001)

print(ids)

print(ytest)

results_cnn=pd.DataFrame({'ImageId':ids, 'Label':ytest})

results_cnn.head()
filename='Digit_Recognizer_Predictions_CNN.csv'

results_cnn.to_csv(filename, index=False)

print('Saved file ',filename)