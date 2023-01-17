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

import matplotlib.pyplot as plt

import keras

from keras.layers import InputLayer , Conv2D, MaxPool2D, BatchNormalization, Activation, Flatten, Dense

from keras.models import Sequential

import tensorflow as tf

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
index = X_test.index

index = index +1

index
print(X_train.shape)

print(X_test.shape)
X_train.head()
Y_train = X_train['label']

X_train.drop('label',axis=1,inplace=True)

X_train.columns
Y_train.head()
m,n = X_train.shape

m1,n1 = X_test.shape

X_train = X_train.values.reshape(m,28,28,1)

X_test = X_test.values.reshape(m1,28,28,1)
print(np.min(X_train),np.max(X_train))
#Let's Normalize 

X_train = X_train/255

X_test = X_test/255
plt.imshow(X_train[3][:,:,0])
plt.imshow(X_test[4][:,:,0])
C = len(Y_train.unique())

Y_train = to_categorical(Y_train,C)

Y_train.shape
# We have 42K training Data. Let's do 60%-40% split

from sklearn.model_selection import train_test_split



Xtr, Xte,Ytr,Yte = train_test_split(X_train,Y_train,test_size=0.4,random_state=2)
#Lets run the Convolution on the train set

model = Sequential([InputLayer(Xtr[0].shape),

            Conv2D(16,kernel_size=(3,3),strides=[1,1],padding='same',kernel_initializer=keras.initializers.glorot_uniform(),name='conv1'),

            Activation('relu'),

            MaxPool2D(pool_size=(2, 2)),

            Conv2D(32,kernel_size=(3,3),strides=[2,2],padding='same',kernel_initializer=keras.initializers.glorot_uniform(),name='conv2'),

            Activation('relu'),

            Conv2D(64,kernel_size=(3,3),strides=[2,2],padding='same',kernel_initializer=keras.initializers.glorot_uniform(),name='conv3'),

            Activation('relu'),

            MaxPool2D(pool_size=(2, 2)),

            Flatten(),

            Dense(10,activation='softmax')

          ])
model.summary()
model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
result = model.fit(Xtr,Ytr,batch_size=64,epochs=30,validation_data=(Xte,Yte))
#get the Test data Predesction

Y_pred_test = model.predict(X_test)

#get the original test data values

Y_pred_test1 = np.argmax(Y_pred_test,axis = 1) 
Y_pred_test1
plt.imshow(X_test[2][:,:,0])
plt.imshow(X_test[0][:,:,0])
plt.imshow(X_test[m1-1][:,:,0])
plt.imshow(X_test[m1-2][:,:,0])
submission_df = pd.DataFrame()

submission_df['Imageid'] = index

submission_df['Label'] = Y_pred_test1
submission_df.to_csv('sample_submission.csv',index=False)
print(os.listdir("../working"))

os.chdir(r'../working')

from IPython.display import FileLink

FileLink('sample_submission.csv')