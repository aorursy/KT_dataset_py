# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
import pandas as pd

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
x_train=train['comment_text']
y_train=train.iloc[:,2:8]
x_test=test['comment_text']
import re
clean_data=[]
for sent in x_train:     
    sent=sent.lower()
    sent = re.sub("[^\w]", " ", sent)
    sent = re.sub(r"\d+", " ", sent)
    sent = re.sub(r"\s+", " ", sent)
    clean_data.append(sent)
clean_data = '\n'.join(clean_data)
clean_data=clean_data.split('\n')
type(clean_data)
from nltk.corpus import stopwords
x_data=[]
for sentence in clean_data:
    sentence=sentence.split(' ')
    word =[word for word in sentence if word not in stopwords.words('english')]
    x_data.append(word)

clean_datat=[]
for sent in x_test:
        sent=sent.lower()
        sent = re.sub("[^\w]", " ", sent)
        sent = re.sub(r"\d+", " ", sent)
        sent = re.sub(r"\s+", " ", sent)

        clean_datat.append(sent)
clean_datat = '\n'.join(clean_datat)
clean_datat=clean_datat.split('\n')

test_data=[]
for sentence in clean_datat:
    sentence=sentence.split(' ')
    word =[word for word in sentence if word not in stopwords.words('english')]
    test_data.append(word)

from keras.preprocessing.text import Tokenizer
token=Tokenizer(num_words=200,split=' ')
token.fit_on_texts(x_data)
x_train=token.texts_to_sequences(x_data)
token=Tokenizer(num_words=200,split=' ')
token.fit_on_texts(test_data)
x_tesr=token.texts_to_sequences(test_data)
from keras.preprocessing.sequence import pad_sequences
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test=pad_sequences(x_tesr,maxlen=maxlen)
from keras.models import Sequential

from keras.layers import MaxPool1D

embed_size = 128
model = Sequential()
model.add(Embedding(200, embed_size))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(MaxPool1D(2))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(6, activation="softmax"))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 512
epochs = 2
hist=model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,shuffle=True)
prediction = model.predict(x_test)

print(prediction)
submission = ('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = prediction
submission.to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],'g')
plt.plot(hist.history['val_loss'],'r')

plt.plot(hist.history['acc'],'b')
plt.plot(hist.history['val_acc'],'black')
