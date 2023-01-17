import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
data = pd.read_csv('../input/airways-tweets/Tweets.csv')

data.head()
df = data[['airline_sentiment', 'text']]

df.head()
df['airline_sentiment'].unique()
df['airline_sentiment'].value_counts()
df_pos = df[df['airline_sentiment'] == 'positive']

df_neg = df[df['airline_sentiment'] == 'negative'].iloc[:len(df_pos)]

len(df_pos), len(df_neg)
df = pd.concat([df_pos, df_neg])

df
df = df.sample(len(df))

df
df['review'] = (df['airline_sentiment'] == 'positive').astype('int')

del df['airline_sentiment']

df
import re

token = re.compile('[A-Za-z]+|[,.!?()]')



def regular_text(text):

    new_text = token.findall(text)

    new_text = [word.lower() for word in new_text]

    return new_text
df['text'] = df['text'].apply(regular_text)

df
word_set = set()

for text in df['text']:

    for word in text:

        word_set.add(word)

word_set
len(word_set)
word_list = list(word_set)

# Use 0 to pad the texts, so the index should start at 1.

to_index = dict((word_list[i], i + 1) for i in range(7100))

to_index
# return 0 if the word concerned is not in word_list

train_data = df['text'].apply(lambda x:[to_index.get(word, 0) for word in x])
max(len(x) for x in train_data)
max_len = 40

max_words = 7100 + 1



train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data.values, maxlen=max_len)

train_data.shape
model = keras.Sequential()



model.add(layers.Embedding(max_words, 50, input_length=max_len)) # input_dim, output_dim ...

model.add(layers.LSTM(64))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(train_data, df.review.values, epochs=10, batch_size=128, validation_split=0.2)
acc = history.history['acc']

val_acc = history.history['val_acc']



plt.figure()

plt.plot(range(10), acc, 'b', label='acc')

plt.plot(range(10), val_acc, 'r', label='val_acc')







plt.title('Training & Validation Acc')

plt.xlabel('Epoch')

plt.ylabel('Acc')



plt.legend()

plt.show()
loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure()

plt.plot(range(10), loss, 'b', label='loss')

plt.plot(range(10), val_loss, 'r', label='val_loss')





plt.title('Training & Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.legend()

plt.show()