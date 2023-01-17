# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
Y_train = train["label"]

X_train = train.drop(labels=["label"],axis=1)



X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)

g= sns.countplot(Y_train)
from keras.utils.np_utils import to_categorical

Y_train= to_categorical(Y_train, num_classes=10)

X_train = X_train/255.0

test = test/255.0
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)
from keras.models import  Sequential

from keras.layers import Flatten,Dense, Dropout, MaxPool2D,Conv2D

model= Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(10, activation='softmax'))

from keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer,

 loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=86)
val_loss,val_acc = model.evaluate(x_val,y_val)

print(val_loss,val_acc)
predict = model.predict([test])
test.shape
print(np.argmax(predict[0]))
predictions = np.argmax(predict,axis=1)
df = pd.DataFrame({"ImageId": range(1, 28000 + 1), "Label": predictions})

df.to_csv("my_submission.csv", index=False)