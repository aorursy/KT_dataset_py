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
df=pd.read_csv('../input/fake-news/train.csv')
df.head()
df.dropna(inplace=True)
from keras.layers import Dense,Embedding,LSTM,Bidirectional
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer,one_hot
from keras.preprocessing.sequence import pad_sequences
import nltk
X=df['title']
y=df['label']
print(X.shape)
print(y.shape)
import re
import string
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps=PorterStemmer()

def clean_text(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text=text.lower()
    text=text.strip()
    text=re.sub('Re:','Retweet',text)
    text_token=[t for t in word_tokenize(text) if t not in stopwords.words('english')]
    text=' '.join(text_token)
    text=text.strip()
    return text


df1=df.copy()
df1['title_new']=df1['title'].apply(lambda x: clean_text(x))
messages=df1['title_new'].to_list()
messages[:10]
vocab_size=5000#no of vocab to be considered
length=20#length of sentence

one_hot_repr=[one_hot(word,vocab_size)for word in messages]
one_hot_repr[:10]#words replaced with index of words


#since length is different for each sentence we will make all sentence length same 
sent_length=20
emb_docs=pad_sequences(one_hot_repr,sent_length,truncating='post',padding='post')
emb_docs[:10]
#truncat=cut the sentence if lenth is more than 20
#padd 0 at the end if length is small
#creating MODEL
#dimension or output feature  
dim=64 #embedding layer with feature 64
model=Sequential()
model.add(Embedding(vocab_size,dim,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())
x_final=np.array(emb_docs)
y_final=np.array(df1['label'])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,train_size=0.75,random_state=22)


print(x_final.shape,y.shape)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15,batch_size=64)
pred=model.predict_classes(x_test)
from sklearn.metrics  import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
#lets  try adding dropout since training istooclose to 1
from keras.layers import Dropout
dim=64 #embedding layer with feature 64
model=Sequential()
model.add(Embedding(vocab_size,dim,input_length=sent_length))
model.add(Dropout(0.4))
model.add(LSTM(100))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=32)
pred=model.predict_classes(x_test)
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
#Lets check with user input


text='Manchester United won the Primere League ninteen times ,they are not performing upto their standards nowadays'
lst=[]
text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
text=text.lower()
text=text.strip()
text=re.sub('Re:','Retweet',text)
text_token=[t for t in word_tokenize(text) if t not in stopwords.words('english')]
text=' '.join(text_token)
text=text.strip()
lst.append(text)

user_one_hot = [one_hot(words , vocab_size) for words in lst]
user_one_hot

sent_length=20
emb_text=pad_sequences(user_one_hot,sent_length,truncating='post',padding='post')

emb_text

model.predict_classes(emb_text)

text='ghost are present every where, but you cant see them'
lst=[]
text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
text=text.lower()
text=text.strip()
text=re.sub('Re:','Retweet',text)
text_token=[t for t in word_tokenize(text) if t not in stopwords.words('english')]
text=' '.join(text_token)
text=text.strip()
lst.append(text)

user_one_hot = [one_hot(words , vocab_size) for words in lst]


sent_length=20
emb_text=pad_sequences(user_one_hot,sent_length,truncating='post',padding='post')



model.predict_classes(emb_text)