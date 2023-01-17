import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/nlp-getting-started/train.csv')

df.head()
from sklearn.model_selection import train_test_split



text = df['text']

label = df['target']



train_sentences, test_sentences, train_labels, test_labels = train_test_split(text, label, test_size=0.1, random_state=42 )

print(train_sentences.shape)

print(train_labels.shape)

print(test_sentences.shape)

print(test_labels.shape)
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
vocab_size=10000

embedding_dim=4

max_length=50

trunc_type="post"

oov_token="<OOV>"



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)

train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)



test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_sequences = [ [str(tok) for tok in seq] for seq in test_sequences ]

test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type)

model = keras.Sequential([

    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    keras.layers.Flatten(),

    keras.layers.Dense(10, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
num_epochs=10

history=model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(acc)

plt.plot(val_acc)

plt.show()
plt.plot(loss)

plt.plot(val_loss)

plt.show()
test = pd.read_csv('../input/nlp-getting-started/test.csv')

sentences=test['text']

ids=test['id']

sequences=tokenizer.texts_to_sequences(sentences)

padded=pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

test.shape
predictions=model.predict(padded).flatten()

predictions=list(map(lambda x: 1 if x>0.5 else 0, predictions))



result=pd.DataFrame({'id':ids ,'target':predictions})

result.set_index('id', inplace=True)

result.to_csv('./result.csv')

print(result.shape)

result.head()
