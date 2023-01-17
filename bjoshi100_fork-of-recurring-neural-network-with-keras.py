# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import numpy as np

import pandas as pd
df_train = pd.read_csv('../input/fashion-mnist_train.csv')

df_test = pd.read_csv('../input/fashion-mnist_test.csv')
df_train.head()
df_train.shape
df_test.shape
df_train.label.unique()
x_train = df_train.drop('label',1)

print (x_train.head)
y_train = df_train['label']

print (y_train.head(10))
print (type(x_train))

print (type(y_train))
x_train = x_train.as_matrix()

print (type(x_train))

print (x_train.shape)
y_train = y_train.as_matrix()

print (type(y_train))

print (y_train.shape)
x_train_partial = x_train[:40000]

x_train_val = x_train[40000:]

y_train_partial = y_train[:40000]

y_train_val = y_train[40000:]
from keras.utils.np_utils import to_categorical

y_train_partial = to_categorical(y_train_partial)

y_train_val = to_categorical(y_train_val)
from keras import models

from keras import layers

model = models.Sequential()

model.add(layers.Dense(32,activation='relu',input_shape=(784,)))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['acc'])



history = model.fit(x_train_partial,

                    y_train_partial,

                    epochs=50,

                    batch_size=128,

                    validation_data=(x_train_val, y_train_val))
import matplotlib.pyplot as plt



history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(history_dict['acc']) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')           

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')      

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
df_test.head()
df_test.shape
x_test = df_test.drop('label',1)

y_test = df_test['label']

x_test = x_test.as_matrix()

y_test = y_test.as_matrix()
print (type(x_test))

print (type(y_test))

print (x_test.shape)

print (y_test.shape)
y_test = to_categorical(y_test)
results = model.evaluate(x_test,y_test)

results