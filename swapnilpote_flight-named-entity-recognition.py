!wget https://raw.githubusercontent.com/microsoft/CNTK/master/Examples/LanguageUnderstanding/ATIS/Data/atis.train.ctf
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow.keras as keras

import os

import csv

import json

from sklearn.model_selection import train_test_split
for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PATH = '/kaggle/working'
# Let create dataset for training model i.e. x & y

df = pd.read_csv('/kaggle/working/atis.train.ctf', sep='\t', names=['sentences_no', 'word', 'intent', 'label'])

df.head(25)
# Let clean our word, intent and label values

df['word'] = df['word'].apply(lambda x: x.split('#')[-1].strip())

df['intent'] = df['intent'].apply(lambda x: x.split('#')[-1].strip() if not pd.isna(x) else 0)

df['label'] = df['label'].apply(lambda x: x.split('#')[-1].strip())

df.head()
# Read all the words for creating embedding layer

words = list(df.word.unique())

words.append('pad')

words.append('unk')

print(sorted(words))
print(f"Number of words present are {len(words)}")
word2idx = {v:k for k, v in enumerate(words)}

idx2word = {k:v for k, v in enumerate(words)}
print(word2idx)
print(idx2word)
# Read all the labels for using as prediction value

    

labels = list(df.label.unique())

print(sorted(labels))
print(f"Number of labels present are {len(labels)}")
label2idx = {v:k for k, v in enumerate(labels)}

idx2label = {k:v for k, v in enumerate(labels)}
print(label2idx)
print(idx2label)
grouped = df.groupby('sentences_no')



x = []

y = []



for _, group in grouped:

    x.append(list(group.word.values))

    y.append(list(group.label.values))

    

print(x[:2])

print(y[:2])
# Now time to convert all words & labels into number

x = [[word2idx[word] for word in sent] for sent in x]

y = [[label2idx[l] for l in label] for label in y]



print(x[:2])

print(y[:2])
plt.figure(figsize=(20, 8))

sns.distplot(pd.DataFrame({'length_sentences': [len(arr) for arr in x]}))
x = keras.preprocessing.sequence.pad_sequences(x, maxlen=15, padding='pre', truncating='pre', value=word2idx.get('pad'))

y = keras.preprocessing.sequence.pad_sequences(y, maxlen=15, padding='pre', truncating='pre', value=label2idx.get('O'))



print(x.shape, y.shape)
X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size=0.20, random_state=20)
print(X_train.shape, Y_train.shape)
print(X_valid.shape, Y_valid.shape)
input_lyr = keras.layers.Input((15,))

embedding_lyr = keras.layers.Embedding(len(words), 32, input_length=15)(input_lyr)

bdi_lstm_lyr = keras.layers.Bidirectional(keras.layers.LSTM(15, return_sequences=True))(embedding_lyr)

output_lyr = keras.layers.Dense(len(labels), activation='softmax')(bdi_lstm_lyr)



model = keras.models.Model(inputs=input_lyr, outputs=output_lyr)

model.compile(keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X_train, Y_train, epochs=50, batch_size=512, validation_data=(X_valid, Y_valid), verbose=1)
model.fit(X_train, Y_train, epochs=50, batch_size=512, validation_data=(X_valid, Y_valid), verbose=1)
model.fit(X_train, Y_train, epochs=50, batch_size=512, validation_data=(X_valid, Y_valid), verbose=1)
model.fit(X_train, Y_train, epochs=50, batch_size=512, validation_data=(X_valid, Y_valid), verbose=1)
def prediction(query):

    modified_query = f'BOS {query} EOS' 

    query = [word2idx.get(word, word2idx.get('unk')) for word in modified_query.lower().split()]

    query = keras.preprocessing.sequence.pad_sequences([query], maxlen=15, padding='pre', truncating='pre', value=word2idx.get('pad'))

    pred  = [idx2label.get(pos) for pos in np.argmax(model.predict(query)[0], axis=1)]

    

    return modified_query, pred
t_query = 'Please book a flight from dwarka sector 23 from hauz khaas at 12 PM'

t_query, pred = prediction(t_query)

print(t_query.split())

print(pred)