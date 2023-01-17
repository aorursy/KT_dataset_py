# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Sequential
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.utils import to_categorical
from keras.optimizers import rmsprop

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
train_data.head()
X_train, X_test, y_train, y_test = train_test_split(train_data.iloc[:,1:], train_data.iloc[:,0], test_size=0.20, random_state=42)
to_categorical(y_train.values,10)
X_train = X_train.values.reshape((X_train.shape[0],28,28,1))/255.
X_test = X_test.values.reshape((X_test.shape[0],28,28,1))/255.
y_train= to_categorical(y_train.values,10)
y_test = to_categorical(y_test.values,10)
model = Sequential()
model.add(Conv2D(64,kernel_size=(5,5),activation="relu" ,input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=(2,2),activation="relu", strides=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax", name="output"))
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=10,batch_size=200,validation_data=(X_test,y_test))


test_data = pd.read_csv("../input/test.csv")
x_predict = test_data.values
x_predict = x_predict.reshape(test_data.shape[0],28,28,1)/255.
predictions = model.predict_classes(x_predict, verbose=0)
submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
    "Label": predictions})
submissions.to_csv("DR10.csv", index=False, header=True)
!ls



# model prediction on test data

!ls
