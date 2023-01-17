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
import tensorflow as tf

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models

from tensorflow import keras
train_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
target = train_data['label']

train_vars = train_data.drop(['label'],axis=1)
X_train = train_vars/255

y = target
X_train = X_train.values.reshape(X_train.shape[0],28,28,1)
model = models.Sequential()

model.add(layers.Conv2D(64, (4,4),activation='relu',input_shape=(28,28,1)))

model.add(layers.Dropout(0.2, noise_shape=None, seed=None))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32, (7,7),activation='relu'))

model.add(layers.Dropout(0.2, noise_shape=None, seed=None))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32, (2,2),activation='relu'))

model.add(layers.Dropout(0.2, noise_shape=None, seed=None))



model.add(layers.Flatten())

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))



model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=8)

lr_reduction = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,factor=0.2)
model.fit(X_train,np.array(y),

          epochs=50,validation_split=0.2,

         batch_size=128, shuffle=True,callbacks =[lr_reduction,es])
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

ImageId = test_data['id']

NN_test = test_data.drop(['id'],axis=1)

NN_test = NN_test/255

NN_test = NN_test.values.reshape(NN_test.shape[0],28,28,1)



predictions = model.predict_classes(NN_test)
sub = pd.DataFrame({'id':ImageId, 'label':predictions})

sub.to_csv("submission.csv",index=False)