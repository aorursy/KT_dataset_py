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
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense
train = pd.read_csv("../input/train.csv")

test_input = pd.read_csv("../input/test.csv")
train_x = train.drop(['ID_code', 'target'], axis = 1)

train_y = train['target']
test = test_input.drop(['ID_code'], axis = 1)
ss = StandardScaler()

train_x_scaled = ss.fit_transform(train_x)

test_scaled = ss.transform(test)
encoder = LabelEncoder()

encoder.fit(train_y)

train_y_encoded = encoder.transform(train_y)
from keras.layers import Dropout

model = Sequential()

model.add(Dense(200, activation='relu', kernel_initializer='normal'))

model.add(Dropout(0.4))

# model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
lr = 0.01

from keras.optimizers import SGD

# _opt = SGD(lr)

_opt= 'adam'

_loss = 'binary_crossentropy'



model.compile(loss=_loss, optimizer=_opt, metrics=['accuracy'])
# Early stopping 

from keras.callbacks import EarlyStopping

_es_monitor = 'val_loss'

_es_patience = 50

es = EarlyStopping(monitor=_es_monitor, mode='min', verbose=1, patience=_es_patience)
# Model check point

from keras.callbacks import ModelCheckpoint

_mc_model_location = 'v1_model.h5'

_mc_monitor = 'val_acc'

mc = ModelCheckpoint(_mc_model_location, monitor=_mc_monitor, mode='max', verbose=1, save_best_only=True)
#batch size and number of epchos 

_batch_size = 1

_epochs = 1000
history = model.fit(train_x_scaled, train_y_encoded, validation_split=0.20,

                    epochs=_epochs, batch_size = len(train_x_scaled), verbose=1, callbacks=[es, mc])
metrics = model.evaluate(train_x, train_y_encoded)

print("\n%s: %.2f%%" % (model.metrics_names[1], metrics[1]*100))
# plot the accuracy - Train vs Valid

import matplotlib.pyplot as plt

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.title('Accuracy - Train vs validation')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

import matplotlib.pyplot as plt



# plot loss - Train vs Valid

plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='test loss')

plt.title('Loss - Train vs validation')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
predict = model.predict(test_scaled)

result = pd.DataFrame({"ID_code": pd.read_csv("../input/test.csv")['ID_code'], "target": predict[:,0]})