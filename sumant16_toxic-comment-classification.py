import numpy as np

import pandas as pd

from keras.preprocessing.text import text_to_word_sequence

import nltk

from nltk.corpus import stopwords

from textblob import TextBlob

from textblob import Word

import string

import re

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense,Activation,Dropout,LSTM,Embedding,Input,Flatten,BatchNormalization

from keras.layers import Bidirectional,GlobalMaxPool1D,Conv1D,SimpleRNN,MaxPooling1D
import pandas as pd

sample_submission=pd.read_csv("../input/sample_submission-5.csv")

test= pd.read_csv("../input/test-7.csv")

test_labels = pd.read_csv("../input/test_labels.csv")

train= pd.read_csv("../input/train-6.csv")
train.head()
train.isnull().any()
test.isnull().any()


y=train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

stop=stopwords.words('english')
train['comment_text']=train['comment_text'].apply(lambda x:" ".join(word.lower() for word in str(x).split() if word.lower() not in stop))
test['comment_text']=test['comment_text'].apply(lambda x:" ".join(word.lower() for word in str(x).split() if word.lower() not in stop))
train['comment_text']=train['comment_text'].apply(lambda x:" ".join(Word(word).lemmatize() for word in str(x).split()))

test['comment_text']=test['comment_text'].apply(lambda x:" ".join(Word(word).lemmatize() for word in str(x).split()))

translator = str.maketrans('', '', string.punctuation)
train['comment_text']=train['comment_text'].apply(lambda x:x.translate(translator))
test['comment_text']=test['comment_text'].apply(lambda x:x.translate(translator))
train['comment_text']=train['comment_text'].apply(lambda x:text_to_word_sequence(x))
test['comment_text']=test['comment_text'].apply(lambda x:text_to_word_sequence(x))
MAX_SEQUENCE_LENGTH=300

MAX_NB_WORDS=20000

embedding_dim=100
tokenizer=Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(train['comment_text'])

train_sequences=tokenizer.texts_to_sequences(train['comment_text'])

test_sequences=tokenizer.texts_to_sequences(test['comment_text'])

train_data=pad_sequences(train_sequences,maxlen=MAX_SEQUENCE_LENGTH)

test_data=pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)
word_index=tokenizer.word_index
len(word_index)
model=Sequential()

model.add(Embedding(MAX_NB_WORDS,embedding_dim,input_length=MAX_SEQUENCE_LENGTH))

model.add(Dropout(0.5))

model.add(Conv1D(128,5,activation="relu"))

model.add(MaxPooling1D(5))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Conv1D(128,5,activation="relu"))

model.add(MaxPooling1D(5))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dense(6,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_data,y,batch_size=32,epochs=4,validation_split=0.1)
y_test=model.predict([test_data])

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

sample_submission[list_classes]=y_test

sample_submission
model1=Sequential()

model1.add(Embedding(MAX_NB_WORDS,embedding_dim,input_length=MAX_SEQUENCE_LENGTH))

model1.add(Bidirectional(LSTM(16,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)))

model1.add(Dropout(0.5))

model1.add(Conv1D(128,5,activation="relu"))

model1.add(MaxPooling1D(5))

model1.add(Dropout(0.5))

model1.add(BatchNormalization())

model1.add(Dropout(0.5))

model1.add(Conv1D(128,5,activation="relu"))

model1.add(MaxPooling1D(5))

model1.add(Dropout(0.5))

model1.add(BatchNormalization())

model1.add(Flatten())

model1.add(Dense(128,activation="relu"))

model1.add(Dense(6,activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



#model1.fit(train_data,y,batch_size=32,epochs=4,validation_split=0.1)