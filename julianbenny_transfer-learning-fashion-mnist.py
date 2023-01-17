# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_label),(test_images,test_label)=fashion_mnist.load_data()
print(len(train_images),len(train_label))

print(len(test_images),len(test_label))
train_images.shape
from keras.utils import np_utils
pt_x_train = []

pt_y_train = []

pt_x_test = []

pt_y_test = []



tl_x_train = []

tl_y_train = []

tl_x_test = []

tl_y_test = []



m=60000



for i in range(m):

    if train_label[i] < 5:

        pt_x_train.append(train_images[i] / 255)

        pt_y_train.append(train_label[i])

    else:

        tl_x_train.append(train_images[i] / 255)

        tl_y_train.append(train_label[i])



m2 = 10000



for i in range(m2):

    if test_label[i] < 5:

        pt_x_test.append(test_images[i] / 255)

        pt_y_test.append(test_label[i])

    else:

        tl_x_test.append(test_images[i] / 255)

        tl_y_test.append(test_label[i])

                         

pt_x_train = np.asarray(pt_x_train).reshape(-1,28,28,1)

pt_x_test = np.asarray(pt_x_test).reshape(-1,28,28,1)

pt_y_train = np_utils.to_categorical(np.asarray(pt_y_train))

pt_y_test = np_utils.to_categorical(np.asarray(pt_y_test))



tl_x_train = np.asarray(tl_x_train).reshape(-1,28,28,1)

tl_x_test = np.asarray(tl_x_test).reshape(-1,28,28,1)

tl_y_train = np_utils.to_categorical(np.asarray(tl_y_train))

tl_y_test = np_utils.to_categorical(np.asarray(tl_y_test))



                         

print(pt_x_train.shape,pt_y_train.shape)

print(pt_x_test.shape,pt_y_test.shape)



print(tl_x_train.shape,tl_y_train.shape)

print(tl_x_test.shape,tl_y_test.shape)                        
from keras.models import Sequential,Model

from keras.layers import Conv2D,Dense,Activation,MaxPool2D,Dropout,Flatten
model = Sequential()



# (28,28,1)



model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))

# (24,24,32)



model.add(Conv2D(16,(5,5),activation='relu'))

# (20,20,16)



model.add(MaxPool2D(pool_size=(2,2)))

# (10,10,16)



model.add(Conv2D(8,(3,3),activation='relu'))

# (8,8,8)



model.add(Flatten())



model.add(Dropout(0.4))



model.add(Dense(128,activation='relu'))



model.add(Dense(64,activation='relu'))



model.add(Dense(5,activation='softmax'))



model.summary()
model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
model.fit(pt_x_train,pt_y_train,

         validation_data=(pt_x_test,pt_y_test),

         epochs=10,

         batch_size=100,

         verbose=2,

         shuffle=True)
for layer in model.layers[:5]:

    layer.trainable = False
x = model.layers[4].output

x = Dropout(0.5)(x)

x = Dense(32,activation='relu')(x)

x = Dense(16,activation='relu')(x)

predictions = Dense(10,activation='softmax')(x)
tl_model = Model(model.input,predictions)
tl_model.summary()
tl_model.compile(optimizer='adam',

          loss='categorical_crossentropy',

          metrics=['accuracy'])
tl_model.fit(tl_x_train,tl_y_train,

            validation_data=(tl_x_test,tl_y_test),

            batch_size=100,

            epochs=10,

            verbose=2,

            shuffle=True)