import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
from tensorflow import keras
import re
import string

import os
kaggle_path = '../input/nlp-getting-started/train.csv'
df = pd.read_csv(kaggle_path)

df.head()
df.shape  # We have 7613 rows and 5 columns
df['text'][100]
from nltk.corpus import stopwords

cached_stop_words = stopwords.words("english")

def remove_at_url(my_string):
    new_text = re.sub(r'http\S+', '', my_string)
    new_text = re.sub(r'@\S+','', new_text)
    new_text = ''.join([x for x in new_text if x not in string.punctuation])
    new_text = ' '.join([x for x in new_text.split() if x not in cached_stop_words])
    return new_text


df['text'] = df['text'].map(lambda x: remove_at_url(x))
print(df['text'][12])
print(df['text'][100])
df['target'].value_counts()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
vocab_size = 5000

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index
# print(word_index)
df['seq'] = tokenizer.texts_to_sequences(df['text'])
# df['seq'] = pad_sequences(df['seq'], padding='post')
temp = pad_sequences(df['seq'], padding='post')
temp
df.head()
temp
temp.shape  # Hence there are 7613 entries each of length 27
training_size = 6090

training_sentences = temp[: training_size]
validation_sentences = temp[training_size: ]

labels = np.array(df['target'])

training_labels = labels[: training_size]
validation_labels = labels[training_size: ]
len(training_sentences), len(training_labels)
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 32),
    keras.layers.LSTM(24),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
my_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit(training_sentences, training_labels, epochs=200, 
              validation_data=(validation_sentences, validation_labels),
              callbacks=[my_cb])
print(history.history.keys())
epochs = len(history.history['loss'])
epochs
y1 = history.history['loss']
y2 = history.history['val_loss']
x = np.arange(1, epochs+1)

plt.plot(x, y1, y2)
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
y1 = history.history['acc']
y2 = history.history['val_acc']
x = np.arange(1, epochs+1)

plt.plot(x, y1, y2)
plt.legend(['acc', 'val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.tight_layout()
res = model.evaluate(validation_sentences, validation_labels)
kaggle_test_path = '../input/nlp-getting-started/test.csv'

test_df = pd.read_csv(kaggle_test_path)
test_df.head()
test_seq = test_df['text'].map(lambda x: remove_at_url(x))
test_seq
test_seq = tokenizer.texts_to_sequences(test_seq)
test_seq
test_seq = pad_sequences(test_seq,  padding='post')
test_seq
predictions = model.predict(test_seq)
predictions = predictions.flatten()
predictions
predictions = np.rint(predictions)
predictions = predictions.astype(np.int32)
predictions
ans = pd.DataFrame({'id': test_df['id'], 'target': predictions})
ans.head()
kaggle_output_path = './my_submission.csv'
ans.to_csv(kaggle_output_path, index=False)