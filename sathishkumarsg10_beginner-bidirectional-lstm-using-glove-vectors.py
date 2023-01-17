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
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
word_to_vec_map = {}
words = set()
with open('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt') as file:
    for line in file:
        values = line.strip().split()
        curr_word = values[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.array(values[1:], dtype = np.float64)
len(word_to_vec_map)
train.shape, test.shape, test_labels.shape
train.head(20)
test.head(20)
train.info()
test.info()
train_sentences = train['comment_text'].values
test_sentences = test['comment_text'].values
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = 10000)
from keras.preprocessing.sequence import pad_sequences
max_seq_length = 1000
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_seq_train = pad_sequences(train_sequences, maxlen = max_seq_length)
padded_seq_test = pad_sequences(test_sequences, maxlen = max_seq_length)
index = tokenizer.word_index
len(index)
embedding_matrix = np.zeros((len(index) + 1, 50))
for word, i in index.items():
    temp = word_to_vec_map.get(word)
    if temp is not None:
        embedding_matrix[i] = temp
classes = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
y = train[classes].values
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(padded_seq_train, y, test_size = 0.3, random_state = 21)
yTrain.shape
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, Dense, LSTM, GlobalMaxPooling1D, Dropout
embed_layer = Embedding(len(index) + 1, 50, input_length = max_seq_length, weights = [embedding_matrix] )
model = Sequential()
model.add(embed_layer)
model.add(Bidirectional(LSTM(50, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation = 'sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
history = model.fit(xTrain, yTrain, epochs = 2, batch_size = 128, validation_split = 0.1)
result = model.evaluate(xTest,yTest)
pred = model.predict(padded_seq_test)
pred.shape
sample = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')
sample.head()
test.head()
sample[classes] = pred
sample.head(20)
sample.to_csv('submission.csv', index = False)
