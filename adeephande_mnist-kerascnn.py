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
import tensorflow.keras as keras

import matplotlib.pyplot as plt

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization

from keras.models import Sequential

from keras.utils import to_categorical

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.shape
train.head()

train_1 = train.iloc[:,1:785]

train_1.head()
train_y = train["label"]

train_y.head()

train_n =train_1.to_numpy()



train_yn = train_y.to_numpy()
train_n = train_n/255



train_n = train_n.reshape((train_n.shape[0],28,28,1))
model = Sequential()
model.add(Conv2D(64,(3,3),activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,(5,5),activation = 'relu',strides=True,padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.4))





model.add(Conv2D(128,(3,3),activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(128,(5,5),activation = 'relu',strides=True,padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

          

model.add(Conv2D(256,(4,4),activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.4))



model.add(Dense(10,activation = 'softmax'))
history = model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history =  model.fit(train_n,train_yn,epochs=40,batch_size =32,verbose=0)
model.summary()
plt.figure(figsize=[8,6])

plt.plot(history.history['accuracy'],'b')

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('learning curves')

plt.show()
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.shape
test_n = test.to_numpy()

test_n = test_n / 255

test_n = test_n.reshape((test_n.shape[0],28,28,1))

test_n.shape
preds = model.predict(test_n)

preds = np.asarray([np.argmax(pred) for pred in preds])

preds.shape
predictions = pd.DataFrame(preds).rename(columns={0:"Label"})

predictions.index.names = ["ImageID"]

predictions.index +=1

predictions.head(25)
predictions.shape

predictions.to_csv("Y_Predictions1.csv")