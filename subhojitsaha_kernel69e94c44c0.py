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
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
train
xtrain = train.iloc[:,1:].values
ytrain = train.iloc[:,0].values
xtrain.shape
xtest = test.iloc[:,1:].values
ytest = test.iloc[:,0].values
xtest.shape
import keras as kr

ytrain = kr.utils.to_categorical(ytrain)
ytest = kr.utils.to_categorical(ytest)
xtrain = xtrain/255
xtest = xtest/255
xtrain.shape
model = kr.models.Sequential()
model.add(kr.layers.Dense(128,activation="sigmoid",input_shape =(784,)))
model.add(kr.layers.Dense(128,activation="sigmoid"))
model.add(kr.layers.Dense(64,activation="sigmoid"))
model.add(kr.layers.Dense(10,activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
hist = model.fit(xtrain,ytrain,epochs=20,batch_size=256,validation_split=0.2,shuffle=True)
import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"],c="red")
plt.plot(hist.history["val_accuracy"])
plt.show()
plt.plot(hist.history["loss"],c="red")
plt.plot(hist.history["val_loss"])
plt.show()
xtrain.shape
ytrain.shape


xtrain = xtrain.reshape((-1,28,28,1))
xtest = xtest.reshape((-1,28,28,1))
model = kr.models.Sequential()
model.add(kr.layers.Convolution2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(kr.layers.Convolution2D(64,(3,3),activation="relu"))
model.add(kr.layers.Dropout(0.25))
model.add(kr.layers.MaxPooling2D(2,2))

model.add(kr.layers.Convolution2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(kr.layers.Convolution2D(8,(3,3),activation="relu"))
model.add(kr.layers.Dropout(0.25))

model.add(kr.layers.Convolution2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(kr.layers.Convolution2D(8,(3,3),activation="relu"))
model.add(kr.layers.Dropout(0.25))


model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(100,activation="softmax"))
model.add(kr.layers.Dense(10,activation="softmax"))

model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
hist = model.fit(xtrain,ytrain,epochs=80,batch_size=256,validation_split=0.25,shuffle=True)
import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"],c="red")
plt.plot(hist.history["val_accuracy"])
plt.show()
plt.plot(hist.history["loss"],c="red")
plt.plot(hist.history["val_loss"])
plt.show()


