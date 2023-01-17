# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import re
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Read data
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',
                encoding='latin',
                header=None)
df.head()
# Change column names

df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
df.head()
# Dropping all columns but text

df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)
# Replacing 4 with 1 for clarity
# 0:Negative, 1:Positive
df = df.replace(4, 1)
df.head(-5)
# Confirming that the dataset is balanced
sen_value = df['sentiment'].value_counts()
print(sen_value)
plt.bar(sen_value.index, sen_value.values)
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess(text):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
        tokens.append(token)
  return " ".join(tokens)
df['text'] = df['text'].apply(lambda x: preprocess(x))
df['text'][400000]
df = shuffle(df)
embed_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size= 160000 
test_portion=.1
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])

word_index = tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences,
                      maxlen=max_length,
                      padding=padding_type,
                      truncating=trunc_type)

split = int(test_portion * training_size)

test_seq = padded[:split]
train_seq = padded[split:training_size]
test_labels = df['sentiment'][:split]
train_labels = df['sentiment'][split:training_size]
print(vocab_size)
print(word_index['good'])
embed_index = {}

with open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embed_index[word] = coefs;
        
embed_matrix = np.zeros((vocab_size+1, embed_dim));
for word, i in word_index.items():
    embed_vector = embed_index.get(word);
    if embed_vector is not None:
        embed_matrix[i] = embed_vector;
print(len(embed_matrix))
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embed_dim, input_length=max_length, weights=[embed_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.summary()
num_epochs = 50

train_padded = np.array(train_seq)
train_labels = np.array(train_labels)
test_padded = np.array(test_seq)
test_labels = np.array(test_labels)

history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=1)

print("Training Complete")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()


# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()
