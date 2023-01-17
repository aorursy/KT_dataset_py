import pandas as pd
import numpy as np
import nltk
train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
train_data = train['text']
train_target = train['target']
train_data.head()
train_target.head()
train_target.value_counts()
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data.values)
len(tokenizer.word_index)
sequence = tokenizer.texts_to_sequences(train_data.values)
len(sequence) == train_data.shape[0]
from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_seq = pad_sequences(sequences=sequence, maxlen=100, padding='post')
pad_seq.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(pad_seq, train_target.values, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train,y_train)
model.score(x_test,y_test)
x_train.shape
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional

model = Sequential()
model.add(Embedding(len(tokenizer.word_index), 10, input_length=100))
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(100)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train,
         epochs=10,
         batch_size=32)
model.save('nlp.h5')
test.head()
seq = tokenizer.texts_to_sequences(test['text'].values)
pad_seq = pad_sequences(seq, padding='post', maxlen=100)
predict = model.predict(pad_seq)
p = list()
for i in predict:
    if i >=0.5:
        p.append(1)
    else:
        p.append(0)
sub = pd.DataFrame({'id':test['id'], 'target':p})
sub.head()
sub.to_csv('submission.csv')
