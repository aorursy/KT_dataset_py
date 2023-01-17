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
import keras as kr
import matplotlib.pyplot as plt
train=pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv").values
test=pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv").values
x_train=train[:,1:]
y_train=train[:,0]
x_test=test[:,1:]
y_test=test[:,0]
x_train.shape
x_test.shape
y_train=kr.utils.to_categorical(y_train)
y_test=kr.utils.to_categorical(y_test)
import matplotlib.pyplot as plt
plt.imshow(x_train[1].reshape(28,28))

y_train[1]
y_train[1]
model=kr.models.Sequential()
model.add(kr.layers.Dense(512,activation="sigmoid",input_shape=(784,)))
model.add(kr.layers.Dense(512,activation="sigmoid"))
model.add(kr.layers.Dense(512,activation="sigmoid"))
model.add(kr.layers.Dense(10,activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
hist=model.fit(x_train,y_train,epochs=20,batch_size=256,validation_split=0.2,shuffle=True)
import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"],c="red")
plt.plot(hist.history["val_accuracy"])
plt.show()
plt.plot(hist.history["loss"],c="red")
plt.plot(hist.history["val_loss"])
plt.show()


x_train=x_train.reshape((-1,28,28,1))
#ytrain=kr.utils.to_categorical(ytrain)
print(x_train.shape,y_train.shape)
for i in range(10):
  plt.imshow(x_train[i].reshape(28,28),cmap="gray")
  plt.show()
model=kr.models.Sequential()
model.add(kr.layers.Convolution2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(kr.layers.Convolution2D(64,(3,3),activation="relu"))

model.add(kr.layers.Dropout(0.25))
model.add(kr.layers.MaxPooling2D(2,2))

model.add(kr.layers.Convolution2D(32,(5,5),activation="relu",input_shape=(28,28,1)))
model.add(kr.layers.Convolution2D(8,(5,5),activation="relu"))

model.add(kr.layers.Dropout(0.25))

model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(100,activation="relu"))
model.add(kr.layers.Dense(10,activation="relu"))
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
hist=model.fit(x_train,y_train,epochs=40,shuffle=True,batch_size=256,validation_split=0.25)
import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"],c="red")
plt.plot(hist.history["val_accuracy"])
plt.show()
plt.plot(hist.history["loss"],c="red")
plt.plot(hist.history["val_loss"])
plt.show()
