import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, load_model

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Bidirectional

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.layers import Dropout

import re

from nltk.corpus import stopwords

from nltk import word_tokenize

STOPWORDS = set(stopwords.words('english'))

from bs4 import BeautifulSoup

import plotly.graph_objs as go

import chart_studio.plotly as py

import cufflinks

from IPython.core.interactiveshell import InteractiveShell

import plotly.figure_factory as ff

InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
df = pd.read_csv("uci-news-aggregator.csv")
df.head()
df2 = df[["TITLE", "CATEGORY"]]
df2["CATEGORY"].value_counts()
#I do aspire here to have balanced classes

num_cat = 45000

shuff = df2.reindex(np.random.permutation(df.index))
e = shuff[shuff["CATEGORY"] == "e"][:num_cat]

b = shuff[shuff["CATEGORY"] == "b"][:num_cat]

t = shuff[shuff["CATEGORY"] == "t"][:num_cat]

m = shuff[shuff["CATEGORY"] == "m"][:num_cat]
conc = pd.concat([e,b,t,m], ignore_index=True)

conc = conc.reindex(np.random.permutation(conc.index))

conc["LABEL"] = 0
#one hot encoded

conc.loc[conc["CATEGORY"] == "e", "LABEL"] = 0

conc.loc[conc["CATEGORY"] == "b", "LABEL"] = 1

conc.loc[conc["CATEGORY"] == "t", "LABEL"] = 2

conc.loc[conc["CATEGORY"] == "m", "LABEL"] = 3
print(conc["LABEL"][:10])
labels = to_categorical(conc["LABEL"], num_classes = 4)

print(labels[:10])
conc.keys()
conc.drop("CATEGORY", axis = 1)

n_most_common_words = 8000

max_len = 130

tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(conc['TITLE'].values)

sequences = tokenizer.texts_to_sequences(conc['TITLE'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



X = pad_sequences(sequences, maxlen=max_len)
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.20, random_state = 42)
epochs = 10

emb_dim = 128

batch_size = 256
print((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
X.shape[1]
model = Sequential()

model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.7))

model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])
acc = model.evaluate(X_test, Y_test)

print("Loss:",acc[0])

print("Accuracy:",acc[1])
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training Accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
txt = ["Regular fast food eating linked to fertility issues in women"]

seq = tokenizer.texts_to_sequences(txt)

padded = pad_sequences(seq, maxlen=max_len)

pred = model.predict(padded)

labels = ['entertainment', 'bussiness', 'science/tech', 'health']

print(pred, labels[np.argmax(pred)])