import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow.keras.preprocessing.text  import Tokenizer

from tensorflow.keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import EarlyStopping

path = '/kaggle/input/sms-spam-collection-dataset/spam.csv'

data_raw = pd.read_csv(path,encoding='latin-1')



data = data_raw['v2']

label = data_raw['v1']



sns.countplot(label)

plt.xlabel('label')

plt.title('Spam or not? ')



en = LabelEncoder()

label = en.fit_transform(label)

label.reshape(-1,1)



train_data , test_data, train_label, test_label = train_test_split(data, label , test_size=0.15)
max_words = 1000

max_len = 150



max_words = 1000

max_len = 150

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(train_data)

sequences = tokenizer.texts_to_sequences(train_data)

train_sequences = sequence.pad_sequences(sequences,maxlen=max_len)



test_sequences = tokenizer.texts_to_sequences(test_data)

test_sequences = sequence.pad_sequences(test_sequences,maxlen=max_len)





num_epochs = 20



model = keras.models.Sequential()



model.add(keras.layers.Input(shape=(max_len)))

model.add(keras.layers.Embedding(max_words,50,embeddings_regularizer = keras.regularizers.l2(0.15))) #0.15

model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))

model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer = keras.regularizers.l2(0.0025))) #256 e 0.0025

model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001))) #32 e 0.0010005

#model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(1,activation='sigmoid'))

   





model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()



es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=5,baseline=0.99)





history = model.fit(train_sequences,train_label,batch_size=128, epochs=num_epochs, callbacks=[es], validation_data=(test_sequences, test_label))
acc = history.history['acc']

epochs_ = range(0,num_epochs)



plt.plot(epochs_, acc, label='accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')



plt.title('accuracy vs epochs')

plt.legend()





accr = model.evaluate(test_sequences, test_label)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))