import pandas as pd
import numpy as np
df = pd.read_csv("../input/df_text_eng.csv")
df["state"] = df["state"].map({'successful': 1, 'failed': 0})
df.head()
texts = list(df["blurb"].astype(str).str.split(",")) # Convert the column to string and to list... list of strings
labels = list(df["state"].astype(str).str.split(","))
texts[1:6]
labels[1:6]
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 25
training_samples = 172410
validation_samples = 21551
test_samples = 21551
max_words = 20000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of labels tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
k = training_samples + validation_samples
x_val = data[training_samples: k]
y_val = labels[training_samples: k]
l = training_samples + validation_samples + test_samples
x_test = data[k:l]
y_test = labels[k:l]
print('Shape of x_train tensor:', x_train.shape)
print('Shape of x_val tensor:', x_val.shape)
print('Shape of x_test tensor:', x_test.shape)
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM

embedding_dim = 100

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(LSTM(100, recurrent_dropout = 0.3, dropout = 0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(x_train, y_train,
epochs=3,
batch_size=64,
validation_data=(x_val, y_val))