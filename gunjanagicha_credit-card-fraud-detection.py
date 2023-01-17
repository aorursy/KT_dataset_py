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
df=pd.read_csv("../input/creditcard.csv")
df.head()
df.shape
x= df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x
from sklearn.preprocessing import StandardScaler
x= StandardScaler().fit_transform(x)
x.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, random_state=1)
from keras.models import Sequential
from keras.layers import Dense, Dropout
model=Sequential([
        Dense(16, activation='relu', kernel_initializer='he_normal', input_dim=(30)),
        Dense(18, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.25),
        Dense(20, activation='relu', kernel_initializer='he_normal'),
        Dense(24, activation='relu', kernel_initializer='he_normal'),
        Dense(1, activation='sigmoid', kernel_initializer='he_normal')])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=15, epochs=2)
score=model.evaluate(x_test, y_test, batch_size=15)
score[1]*100
