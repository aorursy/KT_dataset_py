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
# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import plot_model

import numpy as np



# Importing data

data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')



del data['Unnamed: 32']

data.head()
data.isnull().sum()
x=data.iloc[:, 2:].values

y=data.iloc[:,1].values



print(x)

print(y)
# encoding binary data(y)

from sklearn.preprocessing import LabelEncoder

labelencoder_y=LabelEncoder()

y=labelencoder_y.fit_transform(y)
# M:1(악성), B:0(양성)
# Splitting the dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)



# Features Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
# Normalizing 

from tensorflow.keras.layers.experimental.preprocessing import Normalization

normalizer=Normalization(axis=-1)

normalizer.adapt(x_train)

normalizer.adapt(x_test)

x_train=normalizer(x_train).numpy()

x_test=normalizer(x_test)
import keras 

from keras.models import Sequential

from keras.layers import Dense, Dropout
model=tf.keras.Sequential()

model.add(tf.keras.layers.Dense(30, input_shape=(30,), activation='relu'))

model.add(tf.keras.layers.Dense(20, activation='sigmoid' ))

model.add(tf.keras.layers.Dense(10, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='softmax'))

# compile

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fitting

model.fit(x_train, y_train, batch_size=100, epochs=150, validation_split=0.2)
y_pred=model.predict(x_test)

print(y_pred, y_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)
print("accuracy : %.2f" % (((cm[0][0]+cm[1][1])/57)*100))
cm
# history

hist=model.fit(x_train, y_train, epochs=100, batch_size=256,validation_split=0.2)



import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])

plt.xlabel("the number of epoch")

plt.ylabel("Loss")

plt.show()

plt.plot(hist.history['accuracy'])

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()
model=tf.keras.Sequential()

model.add(tf.keras.layers.Dense(30, input_shape=(30,), activation='relu'))

model.add(tf.keras.layers.Dense(20, activation='sigmoid' ))

model.add(tf.keras.layers.Dense(10, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# compile

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fitting

model.fit(x_train, y_train, batch_size=100, epochs=150, validation_split=0.2)
y_pred=model.predict(x_test)

print(y_pred, y_test)
loss, accuracy = model.evaluate(x_test,y_test)

print("Accuracy :%.2f"% (accuracy*100))
# history

hist=model.fit(x_train, y_train, epochs=100, batch_size=256,validation_split=0.2)
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])

plt.xlabel("the number of epoch")

plt.ylabel("Loss")

plt.show()
plt.plot(hist.history['accuracy'])

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()
# threshold를 정해주어 결과를 1과0으로 표시

y_pred[y_pred>=0.5]=1

y_pred[y_pred<0.5]=0

print(y_pred, y_test)