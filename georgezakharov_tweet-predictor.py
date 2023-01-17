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
import tensorflow
print(tensorflow.__version__)
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

from tensorflow.keras.callbacks import EarlyStopping

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', halt_on_error=False)
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
train_data.info()
train_data.head()
train_data['target'].value_counts()
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.bar(train_data['target'].value_counts().index, train_data['target'].value_counts())
plt.xlabel('Real and not real tweets groups')
plt.ylabel('Number of targets')

plt.subplot(122)
plt.bar(train_data['target'].value_counts().index, train_data['target'].value_counts(normalize=True))
plt.xlabel('Normalized values count')
plt.ylabel('Real and not real tweets groups')

plt.suptitle('Distribution of target')
plt.show()
all_words = []
for sent in train_data['text']:
    tokenize_word = word_tokenize(sent)
    for word in tokenize_word:
        all_words.append(word)
unique_words = set(all_words)
print(len(unique_words))
vocab_length = 28000
embedded_sentences = [one_hot(sent, vocab_length) for sent in train_data['text']]
print(embedded_sentences[:3])
# Let's count max sent vector size
word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(train_data['text'], key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))
padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')
print(padded_sentences[:3])
model = Sequential()
model.add(Embedding(vocab_length, 20, input_length=length_long_sentence))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=2)
history = model.fit(padded_sentences, train_data['target'], batch_size=100, epochs=20, verbose=1, validation_split=0.2, callbacks=[early_stopping_callback])

print("\nStop on epoch: ", early_stopping_callback.stopped_epoch)
loss, accuracy = model.evaluate(padded_sentences, train_data['target'], verbose=0)
print('Accuracy: %f' % (accuracy*100))
plt.plot(history.history['acc'], label='Test')
plt.plot(history.history['val_acc'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accurancy')
plt.legend()
plt.show()
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
embedded_sentences_test = [one_hot(sent, vocab_length) for sent in test_data['text']]
print(embedded_sentences_test[:3])
padded_sentences_test = pad_sequences(embedded_sentences_test, length_long_sentence, padding='post')
print(padded_sentences_test[:3])
submission['target'] = model.predict(padded_sentences_test)
submission['target'] = submission['target'].round().astype(int)
submission['target']
submission.to_csv('submission.csv', index=False)
submission.head()