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
data=pd.read_csv('../input/predicting-parole-violators/parole.csv')
data.info()
data.head()
data.violator.value_counts()
data.race.value_counts()
data.crime.value_counts()
data.state.value_counts()
data.info()
data.head()
X_train=data[:420]
X_train.head()
X_test=data[420:]
X_test.head()
X_train_train=X_train.drop(['violator'],axis=1)
X_train_train
Y_train_train=X_train['violator']
Y_train_train
X_test_test=X_test.drop(['violator'],axis=1)
X_test_test
Y_test_test=X_test['violator']
X_train_train['age']=(X_train_train['age']-np.mean(X_train_train['age']))/(np.std(X_train_train['age']))
X_train_train
X_train_train['time.served']=(X_train_train['time.served']-np.mean(X_train_train['time.served']))/(np.std(X_train_train['time.served']))

X_train_train['max.sentence']=(X_train_train['max.sentence']-np.mean(X_train_train['max.sentence']))/(np.std(X_train_train['max.sentence']))

X_train_train['multiple.offenses']=(X_train_train['multiple.offenses']-np.mean(X_train_train['multiple.offenses']))/(np.std(X_train_train['multiple.offenses']))
X_test_test['age']=(X_test_test['age']-np.mean(X_test_test['age']))/(np.std(X_test_test['age']))

X_test_test['time.served']=(X_test_test['time.served']-np.mean(X_test_test['time.served']))/(np.std(X_test_test['time.served']))

X_test_test['max.sentence']=(X_test_test['max.sentence']-np.mean(X_test_test['max.sentence']))/(np.std(X_test_test['max.sentence']))

X_test_test['multiple.offenses']=(X_test_test['multiple.offenses']-np.mean(X_test_test['multiple.offenses']))/(np.std(X_test_test['multiple.offenses']))

X_train_train=np.array(X_train_train)
X_train_train
Y_train_train=np.array(Y_train_train)

X_test_test=np.array(X_test_test)

Y_test_test=np.array(Y_test_test)
Y_train_train.shape
X_train_train.shape
import keras
from keras import layers

from keras import models
model=models.Sequential()

model.add(layers.Dense(16,activation='relu',input_shape=(8,)))

model.add(layers.Dense(16,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['acc'])
history=model.fit(X_train_train,Y_train_train,epochs=20,batch_size=16)
history_dict=history.history
history_dict.keys()
history_dict.values()
history_dict.items()
import matplotlib.pyplot as plt

loss_values=history_dict['loss']

epochs=range(1,21)

plt.plot(epochs,loss_values,'bo',label='Training loss')

plt.title('training and validation loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
acc_values = history_dict['acc']



plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
results=model.evaluate(X_test_test,Y_test_test)
results
model.predict(X_test_test)