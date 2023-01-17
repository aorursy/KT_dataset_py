import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

print("We're using TF", tf.__version__)

import keras

print("We are using Keras", keras.__version__)

import keras.utils

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test_df=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
all_x=train_df.drop('label',axis=1)

all_y=train_df['label']

test_all_x=test_df.drop('label',axis=1)

test_all_y=test_df['label']
x_train,x_test,y_train,y_test=train_test_split(all_x,all_y,test_size=0.2,random_state=1)
print('x train shape: ',x_train.shape)

print('x test shape: ',x_test.shape)
clothes_labels={

 0: "T-shirt/top",

 1: "Trouser",

 2: "Pullover",

 3: "Dress",

 4: "Coat",

 5: "Sandal",

 6: "Shirt",

 7: "Sneaker",

 8: "Bag",

 9: "Ankle boot"

}
c=x_train.head(9)
fig=plt.figure(figsize=(10,10))

for i,j in zip(c.index,range(9)):

    fig.add_subplot(3,3,j+1)

    plt.imshow(x_train.loc[i].values.reshape(28,28),cmap='gray_r')

    plt.title('label: {}'.format(clothes_labels[y_train.loc[i]]))

plt.show
y_train_oh=keras.utils.to_categorical(y_train,10)

y_test_oh=keras.utils.to_categorical(y_test,10)

test_all_y_oh=keras.utils.to_categorical(test_all_y,10)

print(y_train_oh.shape)

print(y_test_oh.shape)

print(test_all_y_oh.shape)
print(y_train_oh[:3], y_train[:3])
x_train=(x_train/255)

x_test=(x_test/255)

test_all_x=(test_all_x/255)
from keras.layers import Dense, Activation

from keras.models import Sequential

from keras import backend as K
K.clear_session()

model=Sequential()

model.add(Dense(256,input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('softmax'))
model.summary()
model.compile(

    loss='categorical_crossentropy', # this is our cross-entropy

    optimizer='adam',

    metrics=['accuracy']  # report accuracy during training

)
model.fit(x_train,

          y_train_oh,

          epochs = 40,

          batch_size = 512,

          validation_data = (x_test, y_test_oh))
model.evaluate(test_all_x,test_all_y_oh)
pred=model.predict_classes(test_all_x)
fig=plt.figure(figsize=(20,10))

for i in range(9):

    fig.add_subplot(3,3,i+1)

    plt.imshow(test_df.drop('label',axis=1).loc[i].values.reshape(28,28),cmap='gray_r')

    plt.title('label: {}, predicted_label: {}'.format(clothes_labels[test_all_y.loc[i]],clothes_labels[pred[i]]))

plt.show
train_df=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test_df=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

all_x=train_df.drop('label',axis=1)

all_y=train_df['label']

test_all_x=test_df.drop('label',axis=1)

test_all_y=test_df['label']

x_train,x_test,y_train,y_test=train_test_split(all_x,all_y,test_size=0.2,random_state=1)
y_train_oh=keras.utils.to_categorical(y_train,10)

y_test_oh=keras.utils.to_categorical(y_test,10)

test_all_y_oh=keras.utils.to_categorical(test_all_y,10)
x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1)

test_all_x = test_all_x.values.reshape(test_all_x.shape[0], 28, 28, 1)
x_train = x_train/255

x_test = x_test/255

test_all_x = test_all_x/255
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout

from keras.layers.advanced_activations import LeakyReLU
K.clear_session()

model=Sequential()

model.add(Conv2D(input_shape=(28,28,1),padding='same',kernel_size=3,filters=16))

model.add(LeakyReLU(0.1))

model.add(Conv2D(padding='same',kernel_size=3,filters=32))

model.add(LeakyReLU(0.1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(padding='same',kernel_size=3,filters=32))

model.add(LeakyReLU(0.1))

model.add(Conv2D(padding='same',kernel_size=3,filters=64))

model.add(LeakyReLU(0.1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))

model.add(Dropout(0.5))

model.add(LeakyReLU(0.1))

model.add(Dense(10))

model.add(Activation('softmax'))
model.summary()
model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)
model.fit(

    x_train,

    y_train_oh,

    batch_size=32,

    epochs=10,

    validation_data=(x_test, y_test_oh),

    verbose=0)
model.evaluate(test_all_x,test_all_y_oh)
pred2=model.predict_classes(test_all_x)
fig=plt.figure(figsize=(20,10))

for i in range(9):

    fig.add_subplot(3,3,i+1)

    plt.imshow(test_df.drop('label',axis=1).loc[i].values.reshape(28,28),cmap='gray_r')

    plt.title('label: {}, predicted_label: {}'.format(clothes_labels[test_all_y.loc[i]],clothes_labels[pred2[i]]))

plt.show