import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Flatten

from sklearn.compose import ColumnTransformer
class Vocabulary:

  def __init__(self,max_words):
    self.max_words = max_words
    self.word2index = {}
    self.word2count = {}
    self.index2word = {}
    self.num_words = 0

  def add_word(self, word):
    if word not in self.word2count:
      # First entry of word into vocabulary
      self.word2count[word] = 1
      self.num_words += 1
    else:
      # Word exists; increase word count
      self.word2count[word] += 1
          
  def add_sentence(self, sentence):
    for word in sentence.split(' '):
      self.add_word(word.lower())

  def consolidate(self):
    self.index2word = {0 : "NULL"}

    sortedList = [k for k, v in sorted(self.word2count.items(), key=lambda item: item[1],reverse=True)]
    for idx in range(1,min(len(sortedList),self.max_words)+1):
      self.index2word[idx] = sortedList[idx-1]
    self.word2index = dict({(value,key) for (key,value) in self.index2word.items()})


  def to_word(self, index):
    return self.index2word[index]

  def to_index(self, word):
    return self.word2index[word]
Fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv",index_col=0,error_bad_lines=False,engine='python')
Real = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv",index_col=0,error_bad_lines=False,engine='python')
Fake["Fake"]=1
Real["Fake"]=0
data = pd.concat([Fake,Real])
data=data.sort_values(by='text')
#data
def sent2list(sent):
  idxList=[word2idx.get(word.lower(),0) for word in sent.split(' ')]
  if 0 in idxList: idxList.remove(0)
  return np.array(idxList)
num_words = 20000

voc = Vocabulary(num_words)

for sentence in data.index:
  voc.add_sentence(sentence)

voc.consolidate()

idx2word = voc.index2word
word2idx = voc.word2index
from sklearn.model_selection import train_test_split

y = data['Fake'].values
X = data.index.values

X = np.array([sent2list(sent) for sent in X])

X_train, X_test, y_train, y_test = train_test_split(X,y)

#[idx2word[idx] for idx in X[0]]
from keras.preprocessing.sequence import pad_sequences

max_words = 50

X_train = pad_sequences(X_train, maxlen = max_words)
X_test = pad_sequences(X_test, maxlen = max_words)
from keras.layers.convolutional import Conv1D,MaxPooling1D

model = Sequential()
model.add(Embedding(num_words+1,75,input_length=max_words))
model.add(LSTM(32, dropout=0.9, return_sequences=True))
model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dropout(0.9))
model.add(Dense(1,activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=512,epochs=15,validation_split=0.2)

model.evaluate(X_test,y_test)
def fake_or_not(y):
  if y>0.75: return print("Fake")
  elif y>0.5: return print("Probably Fake")
  elif y>0.25: return print("Probably True")
  else: return print("True")

from re import sub

def preprocess(sent):
    sent = sub(r'[^\w\s]','',sent)
    return sent2list(sent)
title = 'Facebook Points to ‘Signs of Stability’ as Ad Declines Flatten'
sentTest = preprocess(title)

sentTest = sentTest.reshape(1,sentTest.shape[0])
sentTest
sentTest = pad_sequences(sentTest, maxlen = max_words)

y = model.predict(sentTest)
y, fake_or_not(y)