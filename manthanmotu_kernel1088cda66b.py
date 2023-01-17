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
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
X = train.drop('label', axis=1)
y = train.label
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
type(X_train)
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Model,Sequential


X_train = X_train.values.reshape(-1,28, 28)
X_test = X_test.values.reshape(-1, 28, 28)
X_train=np.expand_dims(X_train,-1)
X_test=np.expand_dims(X_test,-1)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train.shape

i=Input(X_train[0].shape)

x=Conv2D(128,3,activation="relu",padding="same")(i)
x=MaxPool2D((2,2),padding="valid")(x)
x=BatchNormalization()(x)

x=Conv2D(64,5,activation="relu",padding="valid")(x)
x=MaxPool2D((2,2),padding="valid")(x)
x=BatchNormalization()(x)
x=Dropout(0.2)(x)
x=Flatten()(x)
x=Dense(64,activation="relu")(x)
x=Dense(10,activation="softmax")(x)
model=Model(i,x)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=5,batch_size=10,validation_data=(X_test,y_test))
y_train
xt=X.values.reshape(-1,28,28)

ytf.shape
xt.shape

ytf.shape
ytf=tf.reshape(ytf,[42000,10,1])
xt.shape
y_train=y_train.toarray()
x_train=train.loc[:,"pixel0":"pixel783"]
x_train
x_train=x_train/255
x_train=x_train.values.reshape(42000,28,28)
x_train
preds=model.predict(test)
preds.shape
pred=np.argmax(preds,axis=1)
pred
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission['Label'] = pred
submission.to_csv('submission.csv', index=False)
submission.shape
test.shape
test=test/255
test=test.values.reshape(-1,28,28,1)
test.shape
