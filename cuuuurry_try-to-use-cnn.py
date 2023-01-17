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

from tensorflow import keras

import matplotlib.pyplot as plt

print(tf.__version__)
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
#adjust the test data to an appropriate size

test=test.iloc[:,:].values

test=test.reshape(28000,28,28,1)

y=train.label.iloc[:].values
#adjust the train data to an appropriate size 

x=train.drop(["label"],axis=1)

x=x.iloc[:,:].values

x=x.reshape(42000,28,28,1)
x, test= x/255.0,test/255.0
Conv2D=keras.layers.Conv2D

MaxPool2D=keras.layers.MaxPool2D

Dense=keras.layers.Dense

Flatten=keras.layers.Flatten

Dropout=keras.layers.Dropout
model = keras.Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

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

model.add(Dense(128, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer="adam", 

             loss="sparse_categorical_crossentropy",

              metrics=["accuracy"]

             )
model.fit(x,y,epochs=30)
ids=np.arange(28000)+1

predictions=model.predict_classes(test)

output=pd.DataFrame({"ImageId":ids, "Label":predictions})

output.to_csv("submission.csv",index=False)
print(output.head(5))