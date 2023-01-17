# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data=None

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    data=pd.read_csv('/kaggle/input/Dataset_for_Classification.csv')

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data.head()
y=data['Attrition']
data.isnull().sum()
temp=data["BusinessTravel"]

temp.describe()
data.drop(['Attrition'],inplace=True,axis = 1)

data.head()
X=pd.get_dummies(data)
X.describe()
from sklearn.preprocessing import StandardScaler
X_Scaled=StandardScaler().fit_transform(X)
y=pd.get_dummies(y)
X_Scaled.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_Scaled,y, test_size = .2, random_state = 0)


import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout 

from keras.layers import BatchNormalization

# Neural network

model = Sequential()

model.add(Dense(48, input_dim=53, activation='relu'))

model.add(Dropout(0.5))

#model.add(BatchNormalization())

model.add(Dense(12, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(6, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100,batch_size=128)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()