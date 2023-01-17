import numpy as np

import pandas as pd

import json as js

import urllib

import gzip

import nltk



from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split

from gensim.models import KeyedVectors

from nltk.corpus import stopwords



from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers.merge import concatenate

from keras.layers.normalization import BatchNormalization

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
import re

#import sys  



#reload(sys)  

#sys.setdefaultencoding('utf8')





import string

from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text):

    #text = [w.strip() for w in text.readlines()]

    #text.decode('unicode_escape').encode('ascii','ignore')

    text = str(text)

    #text = text.decode("utf8")

    

    text =  text.split()

    words = []

    for word in text:

      exclude = set(string.punctuation)

      word = ''.join(ch for ch in word if ch not in exclude)

      if word in stops:

        continue

      try: 

        words.append(ps.stem(word))

      except UnicodeDecodeError:

        words.append(word)

    text = " ".join(words)

    

    

    return text.lower()





#Process data



stops = set(stopwords.words("english"))



ps = PorterStemmer()

df = pd.read_csv('../input/fake.csv')

df["type"]= df["type"].replace("bs","fake")

df["type"]= df["type"].replace("conspiracy","fake")



df["type"]= df["type"].replace("satire","real")

df["type"]= df["type"].replace("bias","real")

df["type"]= df["type"].replace("hate","real")

df["type"]= df["type"].replace("junksci","real")

df["type"]= df["type"].replace("state","real")



df.type = df.type.map(dict(real=1, fake=0))

df
df = df[1:1000]



X_train, X_test, y_train, y_test = train_test_split(df['title'], df.type, test_size=0.2)



X_cleaned_train = [clean_text(x) for x in X_train]



X_cleaned_test = [clean_text(x) for x in X_test]







X_cleaned_train[0]
from keras.models import Sequential



from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences



kVECTORLEN = 50



model = Sequential()

model.add(Embedding(5000, 500, input_length=50))

model.add(LSTM(125))

model.add(Dropout(0.4))

model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())





#test_sequence



train_sequence = sequence.pad_sequences(train_sequence, maxlen=50)

test_sequence = sequence.pad_sequences(test_sequence, maxlen=50)



history = model.fit(train_sequence, y_train, validation_data=(test_sequence, y_test), epochs=10, batch_size=64)