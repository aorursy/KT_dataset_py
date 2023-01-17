
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence,text
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import  Embedding, SimpleRNN, LSTM,Masking,Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import metrics, regularizers
from keras.optimizers import RMSprop
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_excel("/kaggle/input/shared-tweets-for-turkish-gsm-operators/gsm-tweets.xlsx",dtype=str)
df.describe()
print(df.head())
len(df)
df = df.apply(lambda x: x.astype(str).str.lower())
df.head()
df.isnull().any()
df['tweet'][1294]
df['tweet']=df.tweet.str.replace(r'http[s]?:(?:[a-zA-Z]|[0-9]|[$-_@.& +]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',regex=True)
df['tweet']=df.tweet.str.replace(r'\@\w*\b','',regex=True)
df['tweet']=df.tweet.str.replace(r'\b\d+','',regex=True)
df['tweet']=df.tweet.str.replace(r'\W*\b\w{1,2}\b','',regex=True)
df['tweet']=df['tweet'].str.findall('\w{2,}').str.join(' ')
df['tweet'][1294]
stop_words = stopwords.words('turkish')    
stop_words
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df = shuffle(df)
len(df)
df[df.tweet.duplicated()].count()
df = df.drop_duplicates()
print(len(df))
df.reset_index(drop=True, inplace=True)
df.tags.value_counts()
le = LabelEncoder()
tags = le.fit_transform(df.tags)
# Kelime numalarandırmada en büyük sayı
num_max = 10000
max_len = 15
## The process of enumerating words
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(df.tweet)
print(df.tweet[10])

for item in df.tweet[10].split():    
    print(tok.word_index[item])

cnn_texts_seq = tok.texts_to_sequences(df.tweet)

cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len,padding='post')

# Örnek
print('***************************************************')
print(df.tweet[50])
print(cnn_texts_mat[50])
print('***************************************************')

X_train,X_test,y_train,y_test=train_test_split(cnn_texts_mat,tags,test_size=0.15, random_state=42)
labels = 'X_train', 'X_test'
sizes = [len(X_train), len(X_test)]
explode = (0, 0.1,)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
#epochs_sayisi=5
batch_size=16
# Çıktı gözükmemesi için
verbose=1
validation_split=0.1
max_len=15
vocab_size=10000
def rnn(epoch_sayisi):
    model=Sequential()
    model.add(Embedding(vocab_size,max_len, trainable=True,input_length=max_len))
    model.add(SimpleRNN(128,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(32,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(16,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(4,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001), metrics=['acc'])
    history=model.fit(X_train, y_train, epochs=epoch_sayisi, batch_size=batch_size,
                      verbose=verbose,validation_split=validation_split)
    return history,model
def lstm(epoch_sayisi):
    model=Sequential()
    model.add(Embedding(vocab_size,max_len, trainable=True,input_length=max_len))
    model.add(LSTM(128,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(4,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])
    history=model.fit(X_train, y_train, epochs=epoch_sayisi, batch_size=batch_size,
                      verbose=verbose,validation_split=validation_split)
    return history,model
def bilstm(epoch_sayisi):
    model=Sequential()
    model.add(Embedding(vocab_size,max_len, trainable=True,input_length=max_len))
    model.add(Bidirectional(LSTM(128,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(16,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(4,activation='relu',kernel_regularizer=regularizers.l2(0.01))))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])
    history=model.fit(X_train, y_train, epochs=epoch_sayisi, batch_size=batch_size,
                      verbose=verbose,validation_split=validation_split)
    return history,model
def cnn(epoch_sayisi):
    model=Sequential()
    model.add(Embedding(vocab_size,max_len, trainable=True,input_length=max_len))
    model.add(Conv1D(128,1,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,1,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(32,1,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(16,1,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(4,1,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])
    history=model.fit(X_train, y_train, epochs=epoch_sayisi, batch_size=batch_size,
                      verbose=verbose,validation_split=validation_split)
    return history,model
def test(algoritma,model):
    #Hiç eğitime sokulmamış veri ile test
    test=model.evaluate(X_test, y_test, verbose=0)
    print("\n\nTest %s modelinde test %s: %.2f%% --- %s: %.2f%%" % (algoritma,model.metrics_names[1], test[1]*100,model.metrics_names[0], test[0]*100))

epoch_sayisi=int(input("Eposh sayisini giriniz..:"))

print('''
    Uygulanacak algoritmayı seçiniz:
    - CNN
    - RNN
    - LSTM
    - BiLSTM
    ''')

algoritma=input("Seçim:").lower()
if(algoritma=="cnn"):
    history,model=cnn(epoch_sayisi)
elif(algoritma=="rnn"):
    history,model=rnn(epoch_sayisi)
elif(algoritma=="lstm"):
    history,model=lstm(epoch_sayisi)
elif(algoritma=="bilstm"):
    history,model=bilstm(epoch_sayisi)
print("**** %s modelinde %s epochluk Acc %.2f  Loss %.2f" %(algoritma,epoch_sayisi,history.history['acc'][epoch_sayisi-1]*100,history.history['loss'][epoch_sayisi-1]*100))

rnn_test=test(algoritma,model)


bilstm_acc = history.history['acc']
bilstm_val_acc =history.history['val_acc']
bilstm_loss=history.history['loss']
bilstm_val_loss=history.history['val_loss']
plt.plot(cnn_acc, label='CNN Acc')
plt.plot(rnn_acc, label='RNN Acc')
plt.plot(lstm_acc, label='LSTM Acc')
plt.plot(bilstm_acc, label='BiLSTM Acc')
plt.title('CNN RNN LSTM BiLSTM Accurary')
plt.ylabel('Validation Loss')
plt.xlabel('Epoch Sayısı')
plt.legend(loc="lower left")
plt.show()
plt.plot(cnn_val_acc, label='CNN Val Acc')
plt.plot(rnn_val_acc, label='RNN Val Acc')
plt.plot(lstm_val_acc, label='LSTM Val Acc')
plt.plot(bilstm_val_acc, label='BiLSTM Val Acc')
plt.title('CNN RNN LSTM BiLSTM Validation Accurary')
plt.ylabel('Validation Loss')
plt.xlabel('Epoch Sayısı')
plt.legend(loc="lower left")
plt.show()
