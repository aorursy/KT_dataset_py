# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.isnull().sum()
train.info()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
sns.distplot(train[train['target']==1]['keyword'].value_counts(),label='disaster')
sns.distplot(train[train['target']==0]['keyword'].value_counts(),label='fake')
plt.legend()


sns.distplot(train[train['target']==1]['location'].value_counts(),label='disaster',kde=False)
sns.distplot(train[train['target']==0]['location'].value_counts(),label='fake',kde=False)
plt.legend()
import seaborn as sns
sns.countplot(train.target)
train.keyword.value_counts()
train['keyword']=train['keyword'].replace(np.nan,'fatalities')
train.location.value_counts()
train['location']=train['location'].replace(np.nan,'USA')

train.isnull().sum()
test.isnull().sum()
test['keyword']=test['keyword'].replace(np.nan,'fatalities')
test['location']=test['location'].replace(np.nan,'USA')
test.isnull().sum()
train=train.drop('id',axis=1)
train.columns
train.text= train.keyword+" "+train.location+" "+train.text
test.text= test.keyword+" "+test.location+" "+test.text
train['text'][0]
test.text[0]
train=train.drop(['keyword','location'],axis=1)
test=test.drop(['keyword','location'],axis=1)

print(train.columns)
print(test.columns)
training_text=train.text
training_text=np.array(training_text)
training_text[:5]
testing_text=test.text.to_numpy()
testing_text[:5]
max_length=0
for text in training_text:
    size=len(list(text.split(" ")))
    max_length=max(max_length,size)
print(max_length)
    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model
from tensorflow.keras import layers
from keras import backend as K
dictionary=[]
for sen in training_text:
    for word in sen.split(' '):
        dictionary.append(word)
dictionary=set(dictionary)
print(len(dictionary))
MAXLEN=40
embedding_dimention = 100
VOCAB_SIZE=20000
tokenizer=Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(training_text)
word_index=tokenizer.word_index
print(len(word_index))
index2word = {}
for k, v in word_index.items():
    if v < VOCAB_SIZE:
        index2word[v] = k
    if v > VOCAB_SIZE:
        continue
word2index={}
for k,v in index2word.items():
    word2index[v]=k
training_text=tokenizer.texts_to_sequences(training_text)
testing_text=tokenizer.texts_to_sequences(testing_text)
training_text=pad_sequences(training_text,maxlen=MAXLEN,padding='post',truncating='post')
testing_text=pad_sequences(testing_text,maxlen=MAXLEN,padding='post',truncating='post')
training_text[:5]
training_y=train.target.to_numpy().ravel()
print(training_y[:10])
embedding_index={}
with open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt') as f:
    for line in f:
        value=line.split()
        word=value[0]
        coeff=np.asarray(value[1:],dtype='float32')
        embedding_index[word]=coeff
    f.close()
print('glove loaded')
def embedding_matrix_creater(embedding_dimention, word2index):
    embedding_matrix = np.zeros((len(word2index) + 1, embedding_dimention))
    for word, i in word2index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = embedding_matrix_creater(embedding_dimention, word2index=word2index)
print(embedding_matrix.shape)
model=tf.keras.models.Sequential()
model.add(keras.layers.Embedding(len(word2index)+1,embedding_dimention,input_length=MAXLEN,weights=[embedding_matrix],trainable=False))
model.add(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
model.add(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2))
model.add(layers.Dense(512,activation='relu',kernel_regularizer='l2'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256,activation='relu',kernel_regularizer='l2'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2,activation='softmax'))


model.summary()
tf.keras.utils.plot_model(model)
print(training_text.shape)
print(training_y.shape)
from sklearn.utils import shuffle
training_text,training_y=shuffle(training_text,training_y,random_state=100)
print(training_text.shape)
print(training_y.shape)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy',recall_m,precision_m,f1_m])
model.fit(training_text,training_y,epochs=10)
test_index=test.id.to_numpy()
test_index=pd.DataFrame(test_index,columns=['id'])
test_index.head()
y_predict=model.predict(testing_text)
y=[]
for i in y_predict:
    y.append(np.argmax(i))
print(len(y))
y=np.array(y)
y=pd.DataFrame(y,columns=['target'])
submission=pd.concat([test_index,y],axis=1)
submission
submission.to_csv('file1.csv',index=False)
