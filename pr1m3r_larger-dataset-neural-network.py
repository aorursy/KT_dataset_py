# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt   # Plotting and graphs



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv'

data = pd.read_csv(path, sep=';')

data.head()
bmi = data.weight / np.square(data.height / 100)

bmi.name = 'bmi'

bmi.head()
dataframes = [data, bmi]

data = pd.concat(dataframes, axis=1)

data.head()
target_name = 'cardio'

target = data[target_name]



target.head()
train = data.drop([target_name, 'id'], axis=1)

train.head()
from sklearn.model_selection import train_test_split   # Splitting the training and testing data



train, test_train, target, test_target = train_test_split(train, target, test_size=0.2, shuffle=True)



print(train.info())

print(test_train.info())
mean = np.mean(train)

std = np.std(train)



train = (train-mean)/(std+1e-7)

test_train = (test_train-mean)/(std+1e-7)
train, valid_train, target, valid_target = train_test_split(train, target, test_size=0.2, shuffle=True)
from keras.models import Sequential   # Type of neural network that will be used

from keras.layers import Dense   # Dense layers for the neural network

from keras.layers import Dropout   # Dropout in case we want to prevent overfitting, use after seeing results without

from keras import optimizers   # Adam optimizer will be used



model = Sequential()



model.add(Dense(32, input_dim=train.shape[1], activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.summary()
from sklearn import metrics   # Allows us to view accuracy and other such values



opt = optimizers.Adam(lr=0.001)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(train, target, epochs=100, validation_data=(valid_train, valid_target), batch_size=10)
plt.plot(history.history['accuracy'], label='acc')

plt.plot(history.history['val_accuracy'], label='val_acc')

plt.ylim((0, 1))

plt.legend()
prediction = model.predict(test_train) > 0.5

prediction = (prediction > 0.5) * 1

accuracy_nn = round(metrics.accuracy_score(test_target, prediction) * 100, 2)

print(accuracy_nn)