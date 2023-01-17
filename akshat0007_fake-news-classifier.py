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
X=df.drop('label',axis=1)
X.head()
y=df['label']
df=df.dropna()
df.head(10)
message=df.copy()
message=message.reset_index()
message.head(10)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
len(message)
corpus=[]
for i in range(0,len(message)):
    text=re.sub('[^a-zA-Z]',' ', message['title'][i])
    text=text.lower()
    text=text.split()
    text=[PorterStemmer().stem(word) for word in text if not word in stopwords.words('english')]
    text=' '.join(text)
    corpus.append(text)
    
y=message['label']
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
vocab_size=5000

onehot_representation=[one_hot(words,vocab_size) for words in corpus]

from tensorflow.keras.preprocessing.sequence import pad_sequences
sent_length=20
embedded_docs=pad_sequences(onehot_representation,padding='pre',maxlen=sent_length)
embedding_vector_features=40
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
print(model.summary())
X_final=np.array(embedded_docs)
y_final=np.array(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64,verbose=1)
