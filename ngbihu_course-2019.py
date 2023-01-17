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
#from keras.models import Sequential

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model



# Nhung cai sau nay, minh co dung dau?

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.utils.np_utils import to_categorical 



import keras.backend as K



K.set_image_data_format('channels_last')



import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Y_train = train['label']

X_train = train.drop(labels='label',axis=1)



#Y_test = test['label']

#X_test = test.drop(labels='label',axis=1)



# Normalize

X_train /= 255

test /= 255 # to do what?

#X_test /= 255
# Reshape

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



print(X_train.shape)
# to one_hot 

Y_train = to_categorical(Y_train, num_classes = 10)
print ("number of training examples = " + str(X_train.shape[0]))

#print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

#print ("X_test shape: " + str(X_test.shape))

#print ("Y_test shape: " + str(Y_test.shape))
random_seed = 2

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# the model



def MaiHaiModel(input_shape):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input) 

    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D((2,2), name='max_pool')(X)

    

    X = Flatten()(X)

    X = Dense(10, activation='softmax', name='fc')(X)

    

    model = Model(inputs= X_input, outputs = X, name='HapplyModel')

    

    return model
maihaiModel = MaiHaiModel(X_train.shape[1:]) 
maihaiModel.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
maihaiModel.fit(X_train,Y_train, epochs=1, batch_size=86)
results = maihaiModel.predict(test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
print(results)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)
print(test.shape)