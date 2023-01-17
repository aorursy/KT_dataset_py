# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df_train.head()
df_x=df_train.iloc[:,1:]
df_x
x_train=df_x.to_numpy()
x_train.shape
X=x_train.reshape(42000,28,28)
plt.imshow(X[120])
plt.show()
X=X.reshape(42000,28,28,1)
Y=df_train["label"].to_numpy()
print(Y[120])
Y=to_categorical(Y)
print(Y.shape)
Y
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=3,padding="same",activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=384,kernel_size=3,padding="same",activation="relu"))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(units=3000,activation="relu"))
model.add(BatchNormalization())

model.add(Dense(units=1500,activation="relu"))
model.add(BatchNormalization())

model.add(Dense(units=400,activation="relu"))
model.add(BatchNormalization())

model.add(Dense(units=10,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(xtrain,ytrain,batch_size=128,epochs=8)
model.evaluate(xtest,ytest,batch_size=32)