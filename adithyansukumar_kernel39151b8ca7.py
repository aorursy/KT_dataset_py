import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
df_train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
df_train.head(1000)
df_test.head()
df_train.info()
df_train.describe()
sns.countplot(df_train['sentiment'])
df_train[df_train['selected_text'].isnull()].index
df_test[df_test['text'].isnull()].index
df_train.dropna(inplace=True)
len(df_train)
x=df_train['selected_text']
y=df_train['sentiment']
x.isnull().sum()
y.isnull().sum()
from nltk.corpus import stopwords
import string
    

def text_process(mess):
    
    # Check characters to see if they are in punctuation
 
    nopunc = [char for char in mess if char not in string.punctuation ]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    word_seq=[word for word in nopunc.split() if word.lower() not in stopwords.words('english') and word.lower() not in ['https',':','/','.','com']]
    return word_seq
x=x.apply(text_process)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_len=100
max_words=20000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x)
sequences=tokenizer.texts_to_sequences(x)

text_data=pad_sequences(sequences,maxlen=max_len)
word_index=tokenizer.word_index
text_data.shape
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder=LabelEncoder()
y=encoder.fit_transform(y)
label_data=np_utils.to_categorical(y)

label_data.shape
x_train=text_data[:26000]
y_train=label_data[:26000]
x_test=text_data[26000:]
y_test=label_data[26000:]
f=open('../input/glove-embeddings/glove.840B.300d.txt')
embeddings_index={}
for line in f:
    values=line.split(' ')
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()

embedding_dim=300
embedding_matrix=np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    if i<max_words:
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dropout,Dense,Bidirectional,LSTM,Flatten,Conv1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=max_len))



model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Bidirectional(LSTM(64,return_sequences=True)))

model.add(LSTM(32))
model.add(Dropout(0.5))


model.add(Dense(32))



model.add(Dense(label_data.shape[1],activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False
model.summary()
es=EarlyStopping(monitor='val_loss')
predictions=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,callbacks=[es])
test_feature=df_test['text']
test_feature=test_feature.apply(text_process)
max_len=100
max_words=20000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(test_feature)
test_sequences=tokenizer.texts_to_sequences(test_feature)

test_data=pad_sequences(test_sequences,maxlen=max_len)
len(df_test['sentiment'])

test_label=df_test['sentiment']
test_label=encoder.fit_transform(test_label)
test_label=np_utils.to_categorical(test_label)
word_index=tokenizer.word_index
test_feature.shape
embedding_dim=300
embedding_matrix=np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    if i<max_words:
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=max_len))



model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Bidirectional(LSTM(64,return_sequences=True)))

model.add(LSTM(32))
model.add(Dropout(0.5))


model.add(Dense(32))



model.add(Dense(label_data.shape[1],activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False
test_predictions=model.predict_classes(test_data)
test_predictions=encoder.inverse_transform(test_predictions)
test_predictions
submission=pd.DataFrame()
submission['textID']=df_test['textID']
submission['sentiment']=test_predictions
submission.head()
