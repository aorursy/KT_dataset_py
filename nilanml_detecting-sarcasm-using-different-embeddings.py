import pandas as pd

df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)

df.head()
df = df.drop(['article_link'], axis=1)

df.head()
df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))

df.head()

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential



max_features = 10000

maxlen = 25

embedding_size = 200



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(df['headline']))

X = tokenizer.texts_to_sequences(df['headline'])

X = pad_sequences(X, maxlen = maxlen)

y = df['is_sarcastic']
EMBEDDING_FILE = '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))



embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
model = Sequential()

model.add(Embedding(max_features, embedding_size, weights = [embedding_matrix]))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(40, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 100

epochs = 5

history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

fig.suptitle("Performance of Glove Vectors")

ax1.plot(history.history['acc'])

ax1.plot(history.history['val_acc'])

vline_cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]

ax1.axvline(x=vline_cut, color='k', linestyle='--')

ax1.set_title("Model Accuracy")

ax1.legend(['train', 'test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

ax2.axvline(x=vline_cut, color='k', linestyle='--')

ax2.set_title("Model Loss")

ax2.legend(['train', 'test'])

plt.show()

max_features = 10000

embedding_size = 200



model = Sequential()

model.add(Embedding(max_features, embedding_size))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(40, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 100

epochs = 5

history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

fig.suptitle("Performance of Model without pretrained embeddings")

ax1.plot(history.history['acc'])

ax1.plot(history.history['val_acc'])

vline_cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]

ax1.axvline(x=vline_cut, color='k', linestyle='--')

ax1.set_title("Model Accuracy")

ax1.legend(['train', 'test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

ax2.axvline(x=vline_cut, color='k', linestyle='--')

ax2.set_title("Model Loss")

ax2.legend(['train', 'test'])

plt.show()
EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

embed_size = 300

word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
max_features = 10000

embedding_size = 300



model = Sequential()

model.add(Embedding(max_features, embedding_size))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(40, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 100

epochs = 5

history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

fig.suptitle("Performance of Model with Fasttext embeddings")

ax1.plot(history.history['acc'])

ax1.plot(history.history['val_acc'])

vline_cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]

ax1.axvline(x=vline_cut, color='k', linestyle='--')

ax1.set_title("Model Accuracy")

ax1.legend(['train', 'test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

ax2.axvline(x=vline_cut, color='k', linestyle='--')

ax2.set_title("Model Loss")

ax2.legend(['train', 'test'])

plt.show()