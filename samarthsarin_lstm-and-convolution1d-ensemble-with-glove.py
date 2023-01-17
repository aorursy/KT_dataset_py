import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
from keras.models import Sequential

from keras.layers import LSTM,Dense,GlobalMaxPool1D,Dropout,Embedding,Bidirectional,Flatten,CuDNNLSTM,Convolution1D,MaxPool1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
df.head()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = df[list_classes].values
tqdm.pandas()
f = open('../input/gloveembeddings/glove.6B.50d.txt')

embedding_values = {}

for line in tqdm(f):

    value = line.split(' ')

    word = value[0]

    coef = np.array(value[1:],dtype = 'float32')

    embedding_values[word] = coef
all_embs = np.stack(embedding_values.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

emb_mean,emb_std
x = df['comment_text']
token = Tokenizer(num_words=20000)
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq,maxlen=50)
vocab_size = len(token.word_index)+1

print(vocab_size)
embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, 50))

for word,i in tqdm(token.word_index.items()):

    values = embedding_values.get(word)

    if values is not None:

        embedding_matrix[i] = values
model1 = Sequential()
model1.add(Embedding(vocab_size,50,input_length=50,weights = [embedding_matrix],trainable = False))
model1.add(Bidirectional(CuDNNLSTM(50,return_sequences=True)))

model1.add(GlobalMaxPool1D())
model1.add(Dense(50,activation = 'relu'))

model1.add(Dropout(0.2))
model1.add(Dense(6,activation = 'sigmoid'))
model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model1.fit(pad_seq,y,epochs =3,batch_size=32,validation_split=0.1)
model1.summary()
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
test.head()
x_test = test['comment_text']

test_seq = token.texts_to_sequences(x_test)

test_pad_seq = pad_sequences(test_seq,maxlen=50)
predict1 = model1.predict(test_pad_seq)
sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

sample_submission[list_classes] = predict1

sample_submission.to_csv('submission1.csv', index=False)
model2 = Sequential()

model2.add(Embedding(vocab_size,50,input_length=50,weights = [embedding_matrix],trainable = False))

model2.add(LSTM(50,dropout=0.1,recurrent_dropout=0.1))

model2.add(Dense(50,activation = 'relu'))

model2.add(Dense(6,activation = 'sigmoid'))

model2.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
history = model2.fit(pad_seq,y,epochs =3,batch_size=32,validation_split=0.1)
predict2 = model2.predict(test_pad_seq)

sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

sample_submission[list_classes] = predict2

sample_submission.to_csv('submission2.csv', index=False)
model3 = Sequential()

model3.add(Embedding(vocab_size,50,input_length=50,weights = [embedding_matrix],trainable = False))

model3.add(Convolution1D(9,kernel_size=5,activation='relu'))

model3.add(MaxPool1D(2))

model3.add(Flatten())

model3.add(Dense(50,activation = 'relu'))

model3.add(Dense(6,activation = 'sigmoid'))

model3.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
history = model3.fit(pad_seq,y,epochs =3,batch_size=32,validation_split=0.1)
predict3 = model3.predict(test_pad_seq)

sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

sample_submission[list_classes] = predict3

sample_submission.to_csv('submission3.csv', index=False)
model4 = Sequential()

model4.add(Embedding(vocab_size,50,input_length=50,weights = [embedding_matrix],trainable = False))

model4.add(Convolution1D(18,kernel_size=3,activation='relu'))

model4.add(MaxPool1D(2))

model4.add(Flatten())

model4.add(Dense(50,activation = 'relu'))

model4.add(Dense(6,activation = 'sigmoid'))

model4.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
history = model4.fit(pad_seq,y,epochs =3,batch_size=32,validation_split=0.1)
predict4 = model4.predict(test_pad_seq)

sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

sample_submission[list_classes] = predict4

sample_submission.to_csv('submission4.csv', index=False)
ensemble_prediction = 0.25*predict1+0.25*predict2+0.25*predict3+0.25*predict4

sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

sample_submission[list_classes] = ensemble_prediction

sample_submission.to_csv('submission_ensemble.csv', index=False)