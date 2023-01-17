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
data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
y=data['label']

data.drop('label',axis=1,inplace=True)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(data,y,test_size=0.33,random_state=42)

X_train,X_cv,y_train,y_cv=train_test_split(X_train,y_train,test_size=0.33,random_state=42)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

X_cv=scaler.transform(X_cv)
import keras

from keras.models import Sequential,Model

from keras.layers import Dense,Input
y_train = keras.utils.to_categorical(y_train, num_classes=10)

y_test = keras.utils.to_categorical(y_test, num_classes=10)

y_cv = keras.utils.to_categorical(y_cv, num_classes=10)
k = Input(shape=(784,))

a=Dense(128,activation='relu')(k)

b=Dense(64,activation='relu')(a)

c=Dense(32,activation='relu')(b)

d=Dense(10,activation='softmax')(c)
model = Model(inputs=[k], outputs=[d])
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=20, batch_size=64,validation_data=(X_cv,y_cv))
model.evaluate(X_test,y_test)