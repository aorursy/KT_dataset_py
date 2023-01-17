import keras as k
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import re
import io
import os
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.DataFrame(columns = ['review', 'estimation'])
path = "../input/movie-review/txt_sentoken/txt_sentoken"
pos_reviews = os.listdir(path + '/pos/')
for i in range(len(pos_reviews)):
    with io.open(path+'/pos/'+pos_reviews[i], "r") as f:
        text = f.read().lower()
        df = df.append({'review':text, 'estimation': 1}, ignore_index=True)
        
neg_reviews = os.listdir(path + '/neg/')
for i in range(len(pos_reviews)):
    with io.open(path+'/neg/'+neg_reviews[i], "r") as f:
        text = f.read().lower()
        df = df.append({'review':text, 'estimation': 0}, ignore_index=True)
df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))   
tokenizer = k.preprocessing.text.Tokenizer(num_words=40000, split=' ')
tokenizer.fit_on_texts(df['review'].values)
X = tokenizer.texts_to_sequences(df['review'].values)
X = k.preprocessing.sequence.pad_sequences(X)
sequence_dict = tokenizer.word_index;
Y = df['estimation'].values
output_dim = 30
lstm_units = 30
dropoutLSTM = 0.5
batch_size = 128
epochs = 30
optimizer = k.optimizers.Adam(lr=0.01, decay=0.01)
model = Sequential()
model.add(Embedding(40000, output_dim = output_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(lstm_units, recurrent_dropout = dropoutLSTM))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
history = model.fit(X, Y, validation_split=0.1, batch_size = batch_size, epochs = epochs)
model.save('model.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
embeddings_index = dict();
with open('../input/glove6b/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;
vocab_size = len(sequence_dict);
embeddings_matrix = np.zeros((vocab_size+1, 100));
for word, i in sequence_dict.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
lstm_units = 30
dropoutLSTM = 0.5
batch_size = 64
epochs = 30
optimizer = k.optimizers.Adam(lr=0.01, decay=0.01)
model = Sequential()
model.add(Embedding(embeddings_matrix.shape[0], output_dim = 100, input_length = X.shape[1], weights=[embeddings_matrix], trainable=False))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(lstm_units, recurrent_dropout = dropoutLSTM))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
history = model.fit(X, Y, validation_split=0.1, batch_size = batch_size, epochs = epochs)
model.save('model_GloVe.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
lstm_units1 = 150
lstm_units2 = 100
dropoutLSTM = 0.3
dense_units = 30
batch_size = 32
epochs = 30
optimizer = k.optimizers.Adam(lr=0.01, decay=0.0001, clipnorm=1)
model = Sequential()
model.add(Embedding(embeddings_matrix.shape[0], output_dim = 100, input_length = X.shape[1], weights=[embeddings_matrix], trainable=False))
model.add(LSTM(lstm_units1, recurrent_dropout = dropoutLSTM, return_sequences=True))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(lstm_units2, recurrent_dropout = dropoutLSTM))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
history = model.fit(X, Y, validation_split=0.1, batch_size = batch_size, epochs = epochs)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save('model_improved.h5')