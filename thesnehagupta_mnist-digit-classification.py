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
test= pd.read_csv('../input/mnist-in-csv/mnist_test.csv') 

train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv') 
test = pd.DataFrame(test)

train = pd.DataFrame(train)
tarray = np.zeros((28,28))

testarray = np.zeros((1000,28,28))

for k in range(1000):

    for n in range(1,29):

        st = str(n) + 'x1'

        sp = str(n) + 'x28'

        tarray[n-1]= test.loc[k:k,st:sp]

    testarray[k] = tarray.copy()
testarray.shape
train
trarray = np.zeros((28,28))

trainarray = np.zeros((6000,28,28))

for k in range(6000):

    for n in range(1,29):

        st = str(n) + 'x1'

        sp = str(n) + 'x28'

        trarray[n-1]= train.loc[k:k,st:sp]

    trainarray[k] = trarray.copy()
trainarray.shape
trainarry = np.zeros((6000,1))

for k in range(6000):

    trainarry[k] = train.loc[k:k,'label':'label']

trainarry.shape

testarry = np.zeros((1000,1))

for k in range(1000):

    testarry[k] = test.loc[k:k,'label':'label']
testarry.shape
trainarray = trainarray.astype("float32")

testarray = testarray.astype("float32")

trainarray/=255

testarray/=255


import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.layers import Conv2D,MaxPooling2D

from keras import backend as K
batch_size = 12

num_classes = 10

epoches = 15

testarry =keras.utils.to_categorical(testarry)

trainarry =keras.utils.to_categorical(trainarry)
testarry[0]
trainarray = trainarray.reshape(6000,28,28,1)

testarray =testarray.reshape(1000,28,28,1)
input_shape =(28,28,1)
model = Sequential()



model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape = input_shape))

model.add(Conv2D(64,(3,3),activation = "relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(num_classes,activation = "softmax"))

model.compile(loss = keras.losses.categorical_crossentropy,optimizer = 'Adam',metrics = ["accuracy"])
model.fit(trainarray,trainarry,epochs = epoches,validation_data = (testarray,testarry))
trray = np.zeros((28,28))

ry = np.zeros((6,28,28))

for k in range(1006,1012):

    for n in range(1,29):

        st = str(n) + 'x1'

        sp = str(n) + 'x28'

        trray[n-1]= test.loc[k:k,st:sp]

    ry[k-1006] = tarray.copy()

ry = ry.reshape(6,28,28,1)

model.predict(ry)
import matplotlib.pyplot as plt

plt.imshow(ry.reshape(6,28,28)[5])