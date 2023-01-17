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
import tensorflow as tf

import keras

from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, AveragePooling2D, Input, ZeroPadding2D

from keras.layers import Activation, Flatten, MaxPooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

import keras.backend as K

from keras.utils import to_categorical



K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
#Get the Test and Train data

train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
#Get the Shape for Train and Test data

print('Train data shape: ',train_data.shape)

print('Test data shape: ',test_data.shape)
#Convert Train and Test data to ND array format

train_data_nd = np.array(train_data)

test_data_nd = np.array(test_data)
#Get the Target label from the Training data

y_train = train_data_nd[:,0]

x_train  = train_data_nd[:,1:]
y_train.shape
x_train.shape
#Normalize the training and Test images

X_train = x_train / 255

X_test = test_data_nd / 255
#display an image randomly from training data

imshow(X_train[140].reshape(28,28))
#display an image randomly from test data

imshow(X_test[200].reshape(28,28))
#Define the Lenet - 5 Model

def lnmodel(x_input):

    

    x_input = Input(x_input)

    

    X = ZeroPadding2D((2,2))(x_input)

    X = Conv2D(128, (5,5), strides = (1,1), name = 'conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'mp0')(X)

    

    X = Conv2D(16, (5,5), strides = (1,1), name = 'conv1')(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'mp1')(X)



    X = Conv2D(120, (5,5), strides = (1,1), name = 'conv2')(X)

    X = BatchNormalization(axis = 3, name = 'bn2')(X)

    X = Activation('relu')(X)

    

    X = Flatten()(X)

    X = Dense(84, activation = 'relu', name = 'fc')(X)

    

    X = Dense(10, activation = 'sigmoid', name = 'fc1')(X)

    

    model = Model(inputs = x_input, outputs = X, name = 'lnmodel')

    

    return model

    
X_train_1 = X_train.reshape(X_train.shape[0], 28,28)

X_train_1.shape
X_test_1 = X_test.reshape(X_test.shape[0],28,28)

X_test_1.shape
X_train_in = X_train_1.reshape(X_train_1.shape[0], 28,28,1)

X_test_in = X_test_1.reshape(X_test_1.shape[0], 28,28,1)
#Perform One hot encoding for Target variables

Y_train_oh = to_categorical(np.expand_dims(y_train, axis = 1), num_classes= 10)
Y_train_oh.shape
X_train_in.shape
lnmodel = lnmodel(X_train_in.shape[1:])
lnmodel.summary()
lnmodel.compile(optimizer= keras.optimizers.Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics= ['accuracy'])
k = lnmodel.fit(X_train_1, Y_train_oh, batch_size= 2048, epochs= 100, validation_split= 0.15, validation_batch_size= 512 )
#Get Training and Validation loss

train_loss = k.history['loss']

val_loss = k.history['val_loss']
plt.plot(train_loss)

plt.plot(val_loss)

plt.legend()

plt.show()
help(plt.legend)
Y_pred = lnmodel.predict(X_test_in, batch_size= 2048)
Y_pred.shape
Y = np.argmax(Y_pred, axis= 1)
Y.shape
Pred = pd.DataFrame(columns=['ImageId', 'Label'])

Pred['Label'] = Y

Pred['ImageId'] = Pred.index + 1

Pred.to_csv("Final1.csv", index = False)