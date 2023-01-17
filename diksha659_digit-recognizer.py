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

from keras.layers import Dropout,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

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
# Importing data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

sample = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train.head()
test.head()
sample.head()
print(train.shape)

print(test.shape)
train_y = train.label

train_y.shape
train_x = train.drop(['label'],axis=1)

train_x.shape
#Normalization 

train_x = train_x/255

print(train_x.shape)

test_x = test/255

print(test_x.shape)
train_x.isnull().any().sum()
train_x = train_x.values.reshape(-1,28,28,1)

train_x.shape
test_x = test.values.reshape(-1,28,28,1)

test_x.shape
#Converting labels to one hot matrices

from keras.utils.np_utils import to_categorical

train_y = to_categorical(train_y,10)

train_y.shape
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(20,6))



for i,j in enumerate([0,6999,13999,20999,27999,34999,41999,7999,36999,14999]):

    plt.subplot(2,5,i+1)

    plt.imshow(train_x[j].reshape(28,28))

    plt.title(np.argmax(train_y[j]))

    plt.axis("off")
#Let's split the data into train and test

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(train_x,train_y,test_size=0.1,random_state=2)

print(train_x.shape)

print(train_y.shape)

print(val_x.shape)

print(val_y.shape)
#CNN Model



def modeln(input_shape = (28,28,1),classes=10):

    

    X_input = Input(input_shape)

    

    X = ZeroPadding2D((3,3))(X_input)

    

    X = Conv2D(filters = 32,kernel_size=(5,5),strides=(1,1),padding='same',kernel_initializer= glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3)(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((2,2))(X)

    

    X = Conv2D(filters=32, kernel_size=(5,5), strides=(1,1),padding='same',kernel_initializer= glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3)(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer= glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3)(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((2,2))(X)

  

    X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer= glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3)(X)

    X = Activation('relu')(X)

    

    X = Flatten()(X)

    X = Dense(256, activation='relu')(X)

    X = Dense(128, activation='relu')(X)

    X = Dense(10, activation='softmax')(X)

    

    #create model

    model = Model(inputs = X_input, outputs=X)

    

    return model

    

    
model = modeln(input_shape=(28,28,1), classes=10)
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(train_x,train_y,epochs=5,batch_size=32,validation_data=(val_x,val_y))
print("loss: "+str(history.history["loss"]))

print("accuracy: "+str(history.history["accuracy"]))

print("val loss: "+str(history.history['val_loss']))

print("val accuracy"+str(history.history['val_accuracy']))
#Let's do data augmentation and check accuracy. In order to avoid over-fitting we do data augmentation.

from keras.preprocessing.image import ImageDataGenerator

data_augmentation = ImageDataGenerator(featurewise_center=False, samplewise_center=False,

    featurewise_std_normalization=False, samplewise_std_normalization=False,

    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.1, width_shift_range=0.2,

    height_shift_range=0.1, brightness_range=None, shear_range=0.0, zoom_range=0.1,

    channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,

    vertical_flip=False, rescale=None, preprocessing_function=None,

    data_format=None, validation_split=0.0, dtype=None)
data_augmentation.fit(train_x)
model.fit(data_augmentation.flow(train_x,train_y,batch_size=32),epochs=5,validation_data=(val_x,val_y))
result = model.predict(test_x)

result = np.argmax(result, axis=1)

result = pd.Series(result,name = 'Label')

result
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)

submission.to_csv("Submission.csv",index=False)