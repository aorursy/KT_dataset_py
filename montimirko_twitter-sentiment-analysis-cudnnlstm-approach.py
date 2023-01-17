# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras as keras

from keras.preprocessing.text import Tokenizer, text

from keras.preprocessing.sequence import sequence

import matplotlib.pyplot as plt

import seaborn as sns

from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding="ISO-8859-1")
#train and label data

label, train = data.iloc[:,0], data.iloc[:,5]



plt.subplot()

label.value_counts().plot(kind="barh")



print(train.iloc[1:10])
num_words = 10000

max_detail = 20000

embedding_dim = 300

max_len = 100



tokenizer = Tokenizer(num_words = num_words)

tokenizer.fit_on_texts(train)

sequences = tokenizer.texts_to_sequences(train)

test_X_seq = sequence.pad_sequences(sequences, maxlen=max_len) #devo paddare le varie frasi



label = label/4 # 0 and 4 --> 0 and 1

label.astype(int)

train_data , test_data, train_label, test_label = train_test_split(test_X_seq, label , test_size=0.2)
epochs_num=3



model = keras.models.Sequential()

model.add(keras.layers.Input(shape=(100,)))



model.add(keras.layers.Embedding(num_words,embedding_dim , trainable=True))#, weights=[embedding_matrix]



model.add(keras.layers.Bidirectional(keras.layers.CuDNNLSTM(512,  kernel_initializer='glorot_uniform', return_sequences=True)))

model.add(keras.layers.Bidirectional(keras.layers.CuDNNLSTM(512,  kernel_initializer='glorot_uniform', return_sequences=True)))

model.add(keras.layers.Bidirectional(keras.layers.CuDNNLSTM(256,  kernel_initializer='glorot_uniform', return_sequences=False)))





model.add(keras.layers.Dense(256))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(512))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(256))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))





model.add(keras.layers.Dense(64))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(1,activation='sigmoid'))



#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



model.summary()


history = model.fit(train_data, train_label, epochs=epochs_num,validation_split=0.1, )



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_num =range(0,epochs_num)



plt.plot(epochs_num, acc, 'b', label='Training acc')

plt.plot(epochs_num, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs_num, loss, 'b', label='Training loss')

plt.plot(epochs_num, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
accr = model.evaluate(test_data, test_label)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))