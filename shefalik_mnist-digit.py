# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

train.shape
train.head()
train.columns
y_train=train['label']

y_train.head()
y_train.head()
X_train=train.drop(['label'],axis=1)/255

X_train.info()
test=pd.read_csv('../input/test.csv')

test.shape
test.head()
X_test=test/255

X_test.head()
#X_train = X_train.reshape(X_train.shape[0],28,28)

#for i in range(6, 9):

 #   plt.subplot(330 + (i+1))

  #  plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

   # plt.title(y_train[i]);
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
model = keras.Sequential()

model.add(Dense(784,input_shape=(784,),activation='relu',))

model.add(Dense(392,activation='relu'))

model.add(Dense(196,activation='relu'))

#model.add(Dense(98,))

model.add(Dense(10,activation='softmax'))
model.compile(loss= keras.losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

#y_test = to_categorical(y_test)
model.fit(X_train, y_train, epochs=10)
predictions= model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), 

    "Label": predictions})
submissions.to_csv("mnist_digit_1.csv",index=False, header=True)