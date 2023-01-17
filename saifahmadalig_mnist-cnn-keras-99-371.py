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

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from keras import optimizers,models,layers,Sequential

from keras.layers import Dense, Dropout, Activation, Flatten ,BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D
df=pd.read_csv('../input/digit-recognizer/train.csv')

dft=pd.read_csv('../input/digit-recognizer/test.csv')
X=df.iloc[0:,1:].values

Y=df.iloc[0:,0].values
X=X/255

dft=dft/255

dft=dft.values
from keras.utils import to_categorical

X=X.reshape(-1,28,28,1)

Y=to_categorical(Y,num_classes=10)

dft=dft.reshape(-1,28,28,1)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=2)
model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.3))





model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(10,activation='softmax'))

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

from keras.callbacks import LearningRateScheduler

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

hist=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=16,callbacks=[annealer])

pri=model.predict_classes(dft)

plt.plot(hist.history['loss'], label='MAE (training data)')

plt.plot(hist.history['val_loss'], label='MAE (validation data)')

plt.xlabel('No. epoch')

plt.legend(loc="upper left")

plt.show()
predictions=model.predict(dft)

predict=np.argmax(predictions)

pre=predictions.argmax(axis=1)
submission = pd.Series(pre,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),submission],axis = 1)

submission.to_csv("E:\\kaggle\\data\\MNIST\\final_submission_v1.csv",index=False)

submission.head()