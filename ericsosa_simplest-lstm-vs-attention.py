import pandas as pd

import numpy as np

import re

from sklearn.model_selection import train_test_split
DFin = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', encoding = 'latin',header=None)

DFin.head()
df = DFin.drop([1,2,3,4], axis = 1) 

df.head()
df[0].unique()
df[0]=df[0].replace(4,1)
df[0].unique()
def preproc(text):

  text = re.sub('@[^\s]+','username', text)

  text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower())

  return text
x = np.array(df.iloc[:, 1].apply(preproc).values)

y = np.array(df.iloc[:, 0].values)

x
from keras.preprocessing.text import Tokenizer



tok = Tokenizer()

tok.fit_on_texts(x)
train = tok.texts_to_sequences(x)

#make 20000 the maximum number for tokens

for i, x in enumerate(train):

    for j, a in enumerate(x):

        if a>20000:

            train[i][j] = 20000
#vocab_size = len(tok.word_index) + 1

vocab_size = 20002
tweetwordlen = 42
from keras.preprocessing.sequence import pad_sequences



xend = pad_sequences(train, maxlen = tweetwordlen, padding='post')

xend.shape
x_train, x_test, y_train, y_test = train_test_split(xend, y, test_size=0.2,random_state=11)
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (Dense, Bidirectional, LSTM, Concatenate, 

                                     Dropout, Embedding, GRU, SimpleRNN,

                                     Input, Attention, GlobalMaxPool1D)

from tensorflow.keras.optimizers import Adam



input_size = tweetwordlen

model = Sequential()

model.add(Embedding(vocab_size, 256, input_length=input_size))

model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='Adam', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, batch_size=1024, validation_split=0.2, epochs=3, verbose=1)
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (Dense, Bidirectional, LSTM, Concatenate, 

                                     Dropout, Embedding, GRU, SimpleRNN,

                                     Input, Attention, GlobalMaxPool1D, Flatten)

from tensorflow.keras.optimizers import Adam



input_size = tweetwordlen

inputs = Input(shape=(input_size))

x = Embedding(vocab_size, 256,

                    input_length=input_size)(inputs)

x = Dropout(0.3)(x)

decoder = Dense(256, activation="relu")(x)

decoder = Dropout(0.3)(decoder)

att = Attention(128)([x, decoder])

att = Flatten()(att)

att = Dense(64, activation="relu")(att)

att = Dropout(0.3)(att)

predictions = Dense(1, activation='sigmoid')(att)



model2 = Model(inputs=inputs, outputs=predictions)



model2.compile(loss='binary_crossentropy',

              optimizer='Adam', metrics=['accuracy'])

model2.summary()

history2 = model2.fit(x_train, y_train, batch_size=1024, validation_split=0.2, epochs=3, verbose=1)
import matplotlib.pyplot as plt

import seaborn as sns



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("LSTM Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.show()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("LSTM Accuracy")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()
model.evaluate(x_test, y_test, verbose=1, batch_size=10000)
def decode_sentiment(score):

    return 1 if score>0.5 else 0



scores = model.predict(x_test, verbose=1, batch_size=10000)

y_pred_1d = [decode_sentiment(score) for score in scores]
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt     



cm = confusion_matrix(y_test, y_pred_1d)/320000

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap="Blues")



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')

ax.set_title('LSTM Confusion Matrix') 

ax.xaxis.set_ticklabels(['Positive', 'Negative'])

ax.yaxis.set_ticklabels(['Positive', 'Negative'])
plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])

plt.title("Attention Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.show()
plt.plot(history2.history['accuracy'])

plt.plot(history2.history['val_accuracy'])

plt.title("Attention Accuracy")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()
model2.evaluate(x_test, y_test, verbose=1, batch_size=10000)
def decode_sentiment(score):

    return 1 if score>0.5 else 0



scores2 = model2.predict(x_test, verbose=1, batch_size=10000)

y_pred_2 = [decode_sentiment(score) for score in scores2]
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt     



cm = confusion_matrix(y_test, y_pred_2)/320000

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap="Blues")



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')

ax.set_title('Attention Confusion Matrix')

ax.xaxis.set_ticklabels(['Positive', 'Negative'])

ax.yaxis.set_ticklabels(['Positive', 'Negative'])