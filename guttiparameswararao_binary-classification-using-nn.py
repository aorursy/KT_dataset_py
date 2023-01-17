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
import pandas as pd

import numpy as np
data=pd.read_csv(r'../input/spam-or-not-spam-dataset/spam_or_not_spam.csv')

data.head()

from sklearn.utils import shuffle

data = shuffle(data)
data['label'].value_counts()

text =[] 

  

# Iterate over each row 

for index, rows in data.iterrows(): 

    # Create list for the current row 

    my_list =str(rows.email)

      

    # append the list to the final list 

    text.append(my_list) 

  

# Print the list 

len(text)
label=list(data['label'])
len(label)
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(text)

sequences = tokenizer.texts_to_sequences(text)
x_train=sequences[:2000]

y_train=label[:2000]

x_test=sequences[2000:]

y_test=label[2000:]
maxlen = 20

from keras import preprocessing

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
x_train[0]
import numpy as np

def vectorize_sequences(sequences, dimension=1000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.

    return results



x_train = vectorize_sequences(x_train)

x_test = vectorize_sequences(x_test)
x_train[0]
y_train[0]
y_train = np.asarray(y_train).astype('float32')

y_test = np.asarray(y_test).astype('float32')
y_train[0]
from keras import models

from keras import layers

model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(1000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',

loss='binary_crossentropy',

metrics=['accuracy'])
history = model.fit(x_train,

y_train,

epochs=20,

batch_size=32,

validation_split=0.3)
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('accuracy')

plt.legend()

plt.show()
results = model.evaluate(x_test, y_test)
results
from keras import models

from keras import layers

from keras import regularizers

model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(1000,),kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',

loss='binary_crossentropy',

metrics=['accuracy'])
history_reg1 = model.fit(x_train,

y_train,

epochs=20,

batch_size=32,

validation_split=0.3)
import matplotlib.pyplot as plt

history_dict = history_reg1.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('accuracy')

plt.legend()

plt.show()
results1 = model.evaluate(x_test, y_test)
results1
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(1000,)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',

loss='binary_crossentropy',

metrics=['accuracy'])
history_d = model.fit(x_train,

y_train,

epochs=20,

batch_size=32,

validation_split=0.3)
import matplotlib.pyplot as plt

history_dict = history_d.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('accuracy')

plt.legend()

plt.show()
results2 = model.evaluate(x_test, y_test)
results
results1
results2