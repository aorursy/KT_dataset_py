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
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten                                      
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.__version__
train_df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_df.head()
train_df.shape
x=train_df.drop('label',axis=1)
x
x.shape
y=train_df['label']
y
x=x.values.reshape(-1,28,28,1)
x.shape
plt.imshow(x[100][:,:,0])
y[100]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
x_train = x_train / 255.0
x_test = x_test / 255.0
model=Sequential()
model.add(Convolution2D(32,(5,5),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(Convolution2D(64,(5,5),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Dropout(0.2))
model.add(Convolution2D(128,(3,3),activation='relu',padding='same'))
model.add(Convolution2D(192,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
filepath = os.path.join("./model_v{epoch}.hdf5")

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
callbacks = [checkpoint]
model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_test,y_test),callbacks=callbacks)
model.evaluate(x_test,y_test)
test_df=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_df.head()
test_df=test_df.values.reshape(-1,28,28,1)
test_df.shape
test_df=test_df/255
classifier=load_model('./model_v14.hdf5')
prediction=classifier.predict(test_df)
prediction.shape
pr=classifier.predict_classes(test_df)
sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sub['Label'] = pr
sub.to_csv('submission1.csv',index=False)