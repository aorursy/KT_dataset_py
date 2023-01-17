import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence,text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import  Embedding, LSTM,Bidirectional,SimpleRNN
from keras import metrics, regularizers
from keras.optimizers import RMSprop
from sklearn.utils import shuffle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('/kaggle/input/tweets-of-trump-and-trudeau/tweets.csv')
df=shuffle(df)
df.info()
print(df.head())
df.drop(columns='id',inplace=True)
df.head()
df.describe()
df.status[0]
df = df.apply(lambda x: x.astype(str).str.lower())
df.head()
df.author.unique()
df.isnull().any()
df[df.status.duplicated()].count()
for status in df.status[0:10]:
    print(status)
df['status']=df.status.str.replace(r'http[s]?:(?:[a-zA-Z]|[0-9]|[$-_@.& +]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',regex=True)
df['status']=df.status.str.replace(r'\@\w*\b','',regex=True)
df['status']=df.status.str.replace(r'\#\w*\b','',regex=True)
df['status']=df.status.str.replace(r'\b\d+','',regex=True)
df['status']=df.status.str.replace(r'\W*\b\w{1,2}\b','',regex=True)
df['status']=df['status'].str.findall('\w{2,}').str.join(' ')
for status in df.status[0:40]:
    print(status)
stop_words = stopwords.words('english')    
stop_words
df['status'] = df['status'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print("Minimum kelime sayısı ..:" ,df['status'].str.split().str.len().min())
print("Maximum kelime sayısı ..:" ,df['status'].str.split().str.len().max())
print("Maximum kelime sayısı ..:" ,df['status'].str.split().str.len().mean())
df[df['status'].str.split().str.len()<3]
df[df['status'].str.split().str.len()<3].count()
df.drop(df[df['status'].str.split().str.len()<3].index,inplace=True)
df.reset_index(drop=True, inplace=True)
df.author.value_counts()
le = LabelEncoder()
tags = le.fit_transform(df.author)
num_max = 10000
max_len = 14
## The process of enumerating words
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(df.status)
print(df.status[10])

for item in df.status[10].split():    
    print(tok.word_index[item])
cnn_texts_seq = tok.texts_to_sequences(df.status)
cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len,padding='post')
print('***************************************************')
print(df.status[10])
print(cnn_texts_mat[10])
print('***************************************************')

X_train,X_test,y_train,y_test=train_test_split(cnn_texts_mat,tags,test_size=0.15, random_state=42)
labels = 'X_train', 'X_test'
sizes = [len(X_train), len(X_test)]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')
plt.title("% Oran")
plt.show()
batch_size=8
verbose=1
validation_split=0.1
num_max=10000
model=Sequential()
model.add(Embedding(num_max,max_len, trainable=True,input_length=max_len))
model.add(SimpleRNN(128,activation='relu',kernel_regularizer=regularizers.l2(0.001),return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(64,activation='relu',kernel_regularizer=regularizers.l2(0.001),return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(32,activation='relu',kernel_regularizer=regularizers.l2(0.001),return_sequences=True))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(16,activation='relu',kernel_regularizer=regularizers.l2(0.001),return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(4,activation='relu',kernel_regularizer=regularizers.l2(0.001))))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])
model.summary()
history=model.fit(X_train, y_train, epochs=10, batch_size=batch_size,verbose=verbose,validation_split=validation_split)
model.evaluate(X_test, y_test, verbose=0)
plt.plot(history.history['acc'], label='Acc')
plt.plot(history.history['val_acc'], label='Val Acc')
plt.ylabel('Acc')
plt.xlabel('Epoch Sayısı')
plt.legend(loc="upper left")
plt.show()