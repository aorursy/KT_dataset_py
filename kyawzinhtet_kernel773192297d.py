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
%load_ext tensorboard
%tensorboard --logdir logs
import tensorflow as tf
import pandas
test=pandas.read_csv(os.path.join(dirname,'test.csv'))
train=pandas.read_csv(os.path.join(dirname,'train.csv'))
y=pandas.read_csv(os.path.join(dirname,'sample_submission.csv'))
print(test)
print(train)
print('the train head is \n', train.head())
print(train['label'])
print(type(train))
print(y.head())
print(y)
import tensorflow.keras as keras
model=keras.Sequential()

model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),input_shape=(28,28,1)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(filters=128,kernel_size=(5,5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_set=train.to_numpy()
print(train_set.shape)
train_x=train_set[:,1:]
print(train_x.shape)
train_y=train_set[:,0]
print(train_y.shape)
train_x=train_x.reshape(42000,28,28,1)
train_y=train_y.reshape(42000,1)
print(train_x.shape)
print(train_y.shape)
tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")
model.fit(train_x/255,train_y,epochs=25,validation_split=0.1,callbacks=[tensorboard_callback])
model.summary()
test_set=test.to_numpy().reshape(28000,28,28,1)
test_y=y.to_numpy()
print(test_y[:,1])

model.evaluate(test_set/255,test_y[:,1])
import matplotlib.pyplot as pyplot
tem=test_set[0,:,:]
tem=tem.reshape(28,28)
pyplot.imshow(tem)


j=model.predict(test_set/255)
for i in range(j.shape[0]):
    f=keras.backend.argmax(j,axis=-1)
print(f.shape)
print(f)
print(f[0])

y['Label']=f
pandas.DataFrame(data=y)
y.to_csv(os.path.join('/kaggle/working/','submission1.csv'),index = False, header=True)
%load_ext tensorboard
%tensorboard --logdir logs
