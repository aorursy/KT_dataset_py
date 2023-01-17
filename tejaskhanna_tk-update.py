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
import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization,AveragePooling2D,BatchNormalization

from sklearn.utils import shuffle

from tensorflow.keras.optimizers import Adam

import random
train=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

validation=pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
def prepare_data(features,target=None):

    '''

        here is a helper function to resize training data to (28,28,1)

        and to categorize the target variable

    '''

    #shuffle data first 

    features,target = shuffle(features,target)

    x = features.values.reshape((len(features),28,28,1))

    y = to_categorical(target)

    print(y.shape)

    # normalize

    x = x/255.0

    return (x,y)

   

    
X,y = train.loc[:,train.columns!='label'],train.label

X_val,y_val = validation.loc[:,validation.columns!='label'],validation.label
print("We have {} training examples and {} validation examples.".format(X.shape[0],X_val.shape[0]))

print("X_train: {} | y_train: {} |\nX_val: {} | y_val: {}".format(X.shape,y.shape,X_val.shape,y_val.shape))
x_train,y_train = prepare_data(X,y)

x_val,y_val = prepare_data(X_val,y_val)
print("X_train: {} | y_train: {} |\nX_val: {} | y_val: {}".format(x_train.shape,y_train.shape,x_val.shape,y_val.shape))
model = Sequential()



# LeNet

# Input Image Dimensionns : 28x28x1



# 1. Conv2D - kernel : 11x11x96. strides=4,4;padding-valid

model.add(Conv2D(filters = 6, kernel_size = (5,5),strides=(1, 1),padding = 'valid', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Dropout(0.5))



# 2. Average Pool  -  kernel - 2x2, strides = 2,2

model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='valid'))



# 3. Conv2d  -  kernel - 5x5x16, strides = 1,1

model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



# 4. Average Pool  -  kernel - 2x2, strides = 2,2

model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='valid'))



# # 5. Convolution - kernel - 5x5x120 ,strides = 1,1

# model.add(Conv2D(filters=120,kernel_size=(4,4),strides=(1,1),padding='valid',activation='relu'))

# model.add(BatchNormalization())



model.add(Flatten())

# 6. Fully connected layer - 84 nodes

model.add(Dense(84,activation='relu'))

model.add(BatchNormalization())



# 7. Output layer - 10 nodes

model.add(Dense(10,activation='softmax'))



# Compile 

optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)

model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])
#model.fit(x_train,y_train,epochs=10,validation_data=(x_val,y_val))

h  = model.fit(x_val,y_val,epochs=10,validation_data=(x_train,y_train),batch_size=64)



test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_features = test.iloc[0:,test.columns!='id']

test_features = test_features.values.reshape((len(test_features),28,28,1))

test_features = test_features.astype('float')
test_features = test_features/255.0
predictions = model.predict(test_features)

predictions = np.argmax(predictions,axis=1)

predictions.shape
final_df = pd.DataFrame({'id':test['id'],'label':predictions})

final_df.head()
final_df.to_csv('submission.csv',index=False)