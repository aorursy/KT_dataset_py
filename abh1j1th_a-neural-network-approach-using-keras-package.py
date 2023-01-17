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
import os
print(os.listdir("../input"))
import pandas as pd

dat=pd.read_csv("../input/roboBohr.csv")
dat.head(5)
df=dat.drop(['Unnamed: 0','pubchem_id'],axis=1)
df.shape
X = df.drop(['Eat'], axis = 1)
Y = df['Eat']
y=Y.values
x=X.values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
x_train.shape

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(1000, input_dim=x.shape[1],kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(500,kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(50,kernel_initializer='normal'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model.fit(x_train, y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)
prd=model.predict(x_test[4:6],verbose=1)
prd
y_test[4:6]
