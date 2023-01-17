import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import string
import re
import os

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
MAX_TOKENS = 10000
OUTPUT_LEN = 300
df_true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
df_false = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
df_true["label"] = 1
df_false["label"] = 0

df = pd.concat((df_true, df_false))
df["text"] = df["title"] + " " + df["text"]
del df["title"]
del df["subject"]
del df["date"]
stop = stopwords.words('english')
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub("\[[^]]*\]", "", text)

def remove_links(text):
    return re.sub(r"http\S+", "", text)

def remove_stopwords(text):
    return " ".join(x for x in text.split() if x.lower() not in stop)

def process_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    
    text = text.translate(text.maketrans("", "", string.punctuation))
    text = remove_stopwords(text)
    return text

df["text"] = df["text"].apply(process_text)
df.head()
X_train, X_test, Y_train, Y_test = train_test_split(df["text"], df["label"], test_size=0.2)
vectorizer = TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=OUTPUT_LEN)
ds_train = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
ds_test = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
vectorizer.adapt(ds_train)
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(2, len(voc))))
embeddings_index = {}
with open("../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))
num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word.decode("utf-8"))
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))
model = Sequential()
model.add(Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
))
model.add(LSTM(units=128, return_sequences=True, dropout=0.25))
model.add(LSTM(units=64, dropout=0.1))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
X_train = vectorizer(np.array([[s] for s in X_train])).numpy()
X_test = vectorizer(np.array([[s] for s in X_test])).numpy()

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
lr_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience = 2, verbose=1, factor=0.5, min_lr=0.00001)
history = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test), callbacks=[lr_reduction])

history = model.fit(X_train, Y_train, batch_size=128, initial_epoch=10, epochs=20, validation_data=(X_test, Y_test), callbacks=[lr_reduction])
history = model.fit(X_train, Y_train, batch_size=128, initial_epoch=20, epochs=30, validation_data=(X_test, Y_test), callbacks=[lr_reduction])
model.save("model.h5")
