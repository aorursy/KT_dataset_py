# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train.head()

test.head()
y_label = train['label']

train = train.drop('label',axis =1)

y_label.head()
print(y_label.shape)

print(train.shape)
#normalize

train_n = (train/255)-0.5

test_n = (test/255)-0.5

Xtrain = train_n.values.reshape(-1,28,28,1)

Xtest = test_n.values.reshape(-1,28,28,1)

print(Xtrain.shape)

print(Xtest.shape)
y_dummy = pd.get_dummies(y_label)

y_dummy.shape
from keras.models import Sequential

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(Xtrain,y_dummy,epochs=3,batch_size=64)
final = model.predict(Xtest)

final.shape
final = np.argmax(final,axis=1)

final.shape
submission = pd.DataFrame({'Label':final,'ImageId':pd.Series(range(1,28001))})

submission.head()
submission.to_csv('DigitRecCNN',index=False)