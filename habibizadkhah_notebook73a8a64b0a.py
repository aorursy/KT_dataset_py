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
df = pd.read_csv("../input/proteinhomology/ProteinHomology.csv")

df.head()
len(df)
df.shape
X = df.to_numpy()
df.shape
x = np.array(X[:, 1:])
x
y = np.array(df['type'])
print(y)
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)
x
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

X_train[0]
y_train
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
print(integer_encoded)
from keras.utils import to_categorical
y_train = to_categorical (integer_encoded)
y_test = to_categorical (y_test)

y_train[101]
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(13,)))
model.add(Dense(27, activation='relu'))

model.add(Dense(54, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))

model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=10,epochs=300,verbose=1)

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(13,)))
model.add(Dense(27, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(54, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=10,epochs=200,verbose=1)