import pandas as pd
import numpy as np

train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep='\t')
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
xtrain = train["Phrase"]
ytrain = pd.get_dummies(train["Sentiment"])
xtest = test["Phrase"]
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

print(xtrain.head(5))
Token = Tokenizer(lower = True, num_words = 20000)
Token.fit_on_texts(xtrain)
Token.fit_on_texts(xtest)
xtrain = Token.texts_to_sequences(xtrain)
xtest = Token.texts_to_sequences(xtest)

word_index = Token.word_index

xtrain = sequence.pad_sequences(xtrain, maxlen = 300)
xtest = sequence.pad_sequences(xtest, maxlen = 300)

embedding_index = {}
f = open('../input/glove6b/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coef = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coef
f.close()
vocab = len(word_index) + 1

embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding

model = Sequential()
emb = Embedding(vocab, 100, weights = [embedding_matrix], input_length = 300, trainable = False)
model.add(emb)
model.add(Conv1D(32, 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(32, 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(32, 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(3))
model.add(LSTM(100))
model.add(Dense(5, activation = 'softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs = 10)
Submission = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
Submission['Sentiment'] = model.predict_classes(xtest)
Submission.to_csv("SentimentSubmission.csv", index = False)