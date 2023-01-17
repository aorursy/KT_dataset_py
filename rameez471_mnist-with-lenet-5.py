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
import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense,Flatten,Conv2D,MaxPool2D

from keras.utils.np_utils import to_categorical

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
train=pd.read_csv('../input/digit-recognizer/train.csv')

train.head()
y=train['label']

X=train.drop(['label'],1)
X=np.array(X)

X.shape
y=np.array(y)

y.shape
def plot_images(X,y):

    for i in range(20):

        plt.subplot(5,4,i+1)

        plt.tight_layout()

        plt.imshow(X[i].reshape(28,28),cmap='gray')

        plt.title('Digit:{}'.format(y[i]))

        plt.xticks([])

        plt.yticks([])

    plt.show()
plot_images(X,y)
trainX,valX,trainY,valY=train_test_split(X,y,test_size=0.1,random_state=42)
print(trainX.shape)

print(trainY.shape)
print(valX.shape)

print(valY.shape)
trainX=trainX/255

valX=valX/255
trainX=trainX.reshape(trainX.shape[0],28,28,1)

valX=valX.reshape(valX.shape[0],28,28,1)
trainY=to_categorical(trainY)

valY=to_categorical(valY)
model=Sequential([

    Conv2D(filters=96,kernel_size=(5,5),padding='same',activation='relu',input_shape=(28,28,1)),

    MaxPool2D(strides=2),

    Conv2D(filters=128,kernel_size=(5,5),padding='valid',activation='relu'),

    MaxPool2D(strides=2),

    Conv2D(filters=128,kernel_size=(3,3),padding='valid',activation='relu'),

    MaxPool2D(strides=2),

    Flatten(),

    Dense(256,activation='relu'),

    Dense(128,activation='relu'),

    Dense(10,activation='softmax')

])
model.summary()
adam=tf.optimizers.Adam(lr=5e-4)

model.compile(loss='categorical_crossentropy',

             metrics=['accuracy'],

             optimizer=adam)
reduce_lr=ReduceLROnPlateau(monitor='val_acc',

                           patience=3,

                           varbose=1,

                           factor=0.2,

                           min_lr=1e-4)

history=model.fit(trainX,trainY,

                 steps_per_epoch=len(trainX)/100,

                 epochs=30,

                 validation_data=(valX,valY),

                 callbacks=[reduce_lr])
score=model.evaluate(valX,valY,batch_size=32)

score
EPOCH=range(1,31)

acc=history.history['accuracy']

val_acc=history.history['val_accuracy']

plt.figure(figsize=(10,6))

plt.plot(EPOCH,acc,'b--',label='Training Accuracy')

plt.plot(EPOCH,val_acc,'b',label='Validation Accurayc')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
loss=history.history['loss']

val_loss=history.history['val_loss']

plt.figure(figsize=(10,6))

plt.plot(EPOCH,loss,'b--',label='Training Loss')

plt.plot(EPOCH,val_loss,'b',label="validation Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
test=pd.read_csv('../input/digit-recognizer/test.csv')

test.head()
testX=np.array(test)

testX.shape
testX=testX/255
testX=testX.reshape((testX.shape[0],28,28,1))

testX.shape
testY=model.predict(testX)
testY=np.argmax(testY,axis=1)

testY[:5]
df_out=pd.read_csv('../input/digit-recognizer/sample_submission.csv')

df_out.head()
df_out['Label']=testY

df_out.head()
df_out.to_csv('out.csv',index=False)