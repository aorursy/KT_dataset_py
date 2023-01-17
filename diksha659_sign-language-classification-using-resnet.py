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
import numpy as np

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.initializers import glorot_uniform

import scipy.misc

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

%matplotlib inline



import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)
X_orig = np.load("../input/sign-language-digits-dataset/X.npy")

Y_orig = np.load("../input/sign-language-digits-dataset/Y.npy")
print(X_orig.shape)

print(Y_orig.shape)
#reshape X_orig

X_orig = X_orig.reshape(-1,64,64,1)

X_orig.shape
plt.figure(figsize=(20,6))



for i,j in enumerate([0,205,411,617,823,1030,1237,1444,1650,1858]):

    plt.subplot(2,5,i+1)

    plt.imshow(X_orig[j].reshape(64,64))

    plt.title(np.argmax(Y_orig[j]))

    plt.axis("off")
label_y = []



label_y = [np.where(i==1)[0][0] for i in Y_orig]

count = pd.Series(label_y).value_counts()

print(count)
#lets reorganize



X_organized = np.concatenate((X_orig[204:409,:],

                              X_orig[822:1028,:],

                              X_orig[1649:1855,:],

                              X_orig[1443:1649,:],

                              X_orig[1236:1443,:],

                              X_orig[1855:2062,:],

                              X_orig[615:822,:],

                              X_orig[409:615,:],

                              X_orig[1028:1236,:],

                              X_orig[0:204,:]),axis = 0)
plt.figure(figsize=(20,6))



for i,j in enumerate([0,205,411,617,823,1030,1237,1444,1650,1858]):

    plt.subplot(2,5,i+1)

    plt.imshow(X_organized[j].reshape(64,64))

    plt.title(np.argmax(Y_orig[j]))

    plt.axis("off")
from sklearn.model_selection import train_test_split



train_x,test_x,train_y,test_y = train_test_split(X_orig,Y_orig,test_size=0.2)

print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
def identity_block(X, f, filters):

    """

    Arguments:

    X -- input shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- shape of the middle CONV's window for the main path

    filters -- number of filters in the CONV layers of the main path

    Returns:

    X -- output of shape (n_H, n_W, n_C)

    """

    

    # Filters

    F1, F2, F3 = filters

    

    # Save the input value. 

    X_shortcut = X

    

    # First component

    X = Conv2D(filters = F1, kernel_size =(1,1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

        

    # Second component

    X = Conv2D(filters= F2, kernel_size=(f,f), strides=(1,1),padding='same',kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3)(X)

    X = Activation('relu')(X)



    # Third component

    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3)(X)



    # Final step

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    

    return X
def convolutional_block(X, f, filters,s = 2):

    """

    X -- input shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- shape of the middle CONV's window for the main path

    filters --number of filters

    s -- stride

    Returns:

    X -- output of the convolutional block shape (n_H, n_W, n_C)

    """

    #Filters

    F1, F2, F3 = filters

    

    # Save the input value

    X_shortcut = X

    # First component

    X = Conv2D(F1, (1,1), strides = (s,s), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)



    # Second component

    X = Conv2D(F2, (f,f), strides = (1,1),padding='same',kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)



    # Third component

    X = Conv2D(F3, (1,1), strides = (1,1), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)



    #SHORTCUT PATH

    X_shortcut = Conv2D(F3, (1,1),strides = (s,s),padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis =3)(X_shortcut)



    # Final step

    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)  

    return X
def ResNet50(input_shape = (64, 64, 1), classes = 10):

    """

    input_shape -- shape of the images of the dataset

    classes -- integer, number of classes



    Returns:

    model

    """

    

    # Define the input with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Stage 1

    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = convolutional_block(X, f = 3, filters = [64, 64, 256],s = 1)

    X = identity_block(X, 3, [64, 64, 256])

    X = identity_block(X, 3, [64, 64, 256])



    # Stage 3

    X = convolutional_block(X,f=3,filters=[128,128,512],s=2)

    X = identity_block(X,3,[128,128,512])

    X = identity_block(X,3,[128,128,512])

    X = identity_block(X,3,[128,128,512])

    

    # Stage 4

    X = convolutional_block(X,f=3,filters=[256,256,1024],s=2)

    X = identity_block(X,3,[256,256,1024])

    X = identity_block(X,3,[256,256,1024])

    X = identity_block(X,3,[256,256,1024])

    X = identity_block(X,3,[256,256,1024])

    X = identity_block(X,3,[256,256,1024])



    # Stage 5

    X = convolutional_block(X,f=3,filters=[512,512,2048],s=2)

    X = identity_block(X,3,[512,512,2048])

    X = identity_block(X,3,[512,512,2048])



    # AVGPOOL

    X = AveragePooling2D((2,2),name="avg_pool")(X)



    # output layer

    X = Flatten()(X)

    X = Dense(classes, activation='softmax',kernel_initializer = glorot_uniform(seed=0))(X)

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='ResNet50')



    return model
model = ResNet50(input_shape=(64,64,1),classes=10)
#Compile

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#fit the training set

model.fit(train_x, train_y, epochs = 20, batch_size = 32)
evals = model.evaluate(test_x,test_y)

print("loss" +str(evals[0]))

print("accuracy" +str(evals[1]))
model.summary()