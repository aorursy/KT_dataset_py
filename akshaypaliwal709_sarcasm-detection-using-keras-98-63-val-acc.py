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
df_1 = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)
df_1_wo_link = df_1.drop(['article_link'], axis=1)
df_1_wo_link.head()
df_2 = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)
df_2.shape
df_2_wo_link = df_2.drop(['article_link'], axis=1)

final_df = pd.concat([df_1_wo_link, df_2_wo_link], axis = 0, sort=False)

final_df.head
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(list(final_df['headline']))
sequences = tokenizer.texts_to_sequences(list(final_df['headline']))
word_index = tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))


X = tokenizer.texts_to_sequences(final_df['headline'])
X = pad_sequences(X, maxlen = maxlen)
y = final_df['is_sarcastic']

glove_dir = '/kaggle/input/textembedding'
embeddings_index={}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

#f = open(os.path.join(glove_dir, 'crawl-300d-2M.vec'))

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors' % len(embeddings_index))

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i<max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
from keras.layers import LSTM,Dense,GRU, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import layers

model = Sequential()
model.add(Embedding(max_words, embedding_dim,input_length = maxlen, weights = [embedding_matrix]))

model.add(Flatten())
model.add(Dense(40, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 10
history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation loss ')
plt.show()
from keras.layers import LSTM,Dense,GRU, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import layers

model = Sequential()
model.add(Embedding(max_words, embedding_dim,input_length = maxlen, weights = [embedding_matrix]))
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))
#model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 6
history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation loss ')
plt.show()