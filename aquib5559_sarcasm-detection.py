import json
import numpy as np
import pandas as pd
import requests
srcsm_json = requests.get('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json')
print(srcsm_json.text[0:500])
sentences = []
labels = []
for item in srcsm_json.json():
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
sentences
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(sentences,labels,test_size=0.25,random_state=42)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=10000,oov_token='OOV')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print(word_index)
vocal_size = 10000
oov_tok = '<oov>'
max_length = 100
trunc_type = 'post'
padding_type = 'post'

X_train = tokenizer.texts_to_sequences(X_train)
X_train_padd = pad_sequences(X_train,maxlen=80,padding='post',truncating='post')


X_test = tokenizer.texts_to_sequences(X_test)
X_test_padd = pad_sequences(X_test,maxlen=80,padding='post',truncating='post')
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Bidirectional,Dense,Dropout
embedding_dim = 16
model = Sequential()
model.add(Embedding(vocal_size,embedding_dim,input_length = max_length))
model.add(Bidirectional(LSTM(60)))
model.add(Dropout(0.4))
model.add(Dense(45,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
X_train_padd = np.array(X_train_padd)
X_test_padd = np.array(X_test_padd)
y_train = np.array(y_train)
y_test = np.array(y_test)
model.fit(X_train_padd,y_train,epochs=30,validation_data=(X_test_padd,y_test))
metrices = pd.DataFrame(model.history.history)
import matplotlib.pyplot as plt
%matplotlib inline
metrices[['loss','val_loss']].plot()
metrices[['accuracy','val_accuracy']].plot()
sentence = ["granny starting to fear spiders in the garden might be real","game of thrones season finale showing this sunday night"]
sentence = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sentence,maxlen=max_length,padding=padding_type,truncating=trunc_type)
print(np.argmax(model.predict(padded)))
sentence = ["Even some of the best life lessons we learn are from the most sarcastic quotes we read over the internet or from our dearest friends and family","Although some people find it difficult to understand the hidden meaning of our sarcastic messages, others have no problem in finding the sense of it at all."]
sentence = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sentence,maxlen=max_length,padding=padding_type,truncating=trunc_type)
print(np.argmax(model.predict(padded)))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
