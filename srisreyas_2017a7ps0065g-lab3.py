import numpy as np 
import pandas as pd 
import os
import tensorflow
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re
#from google.colab import files
#uploaded = files.upload()
#wts=files.upload()
#!unzip nnfl-lab-3-nlp.zip
train_df=pd.read_csv('../input/nnfl-lab-3-nlp/nlp_train.csv')
train_df.head()
len(train_df)
num_words = 25000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
print(train_df['tweet'][0])
tokenizer.fit_on_texts(train_df['tweet'].values)
X = tokenizer.texts_to_sequences(train_df['tweet'].values)
print(X[0])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

max_length_of_text = 280
X = pad_sequences(X, maxlen=max_length_of_text)

print(word_index)
print("Padded Sequences: ")
print(X)
print(X[0])
y = train_df['offensive_language']
num_words=25000
n_lstm=128
model=Sequential()
model.add(Embedding(num_words,64,input_length=280))
model.add(LSTM(n_lstm,dropout=0.2,recurrent_dropout=0.3))
model.add(Dense(128,activation='relu'))
model.add(Dense(1))
print(model.summary())
model.compile(loss = 'mean_squared_error', optimizer='adam',metrics = [tensorflow.keras.metrics.RootMeanSquaredError('rmse')])
load_flag=True
if load_flag:
  model.load_weights('../input/weights/model.h5')
batch_size=32
num_epochs=5
fast=False
if fast:
  num_epochs=3
model.fit(X, y, batch_size = batch_size, epochs = num_epochs)
model.save('model.h5')
test_df=pd.read_csv('../input/nnfl-lab-3-nlp/_nlp_test.csv')
num_words = 25000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
print(test_df['tweet'][0])
tokenizer.fit_on_texts(test_df['tweet'].values)
X_test = tokenizer.texts_to_sequences(test_df['tweet'].values)
print(X_test[0])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

max_length_of_text = 280
X_test = pad_sequences(X_test, maxlen=max_length_of_text)

print(word_index)
print("Padded Sequences: ")
print(X_test)
print(X_test[0])
y_test=model.predict(X_test)
test_df['offensive_language']=y_test
test_df.head()
df=test_df
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)
create_download_link(df)
