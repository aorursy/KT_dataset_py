# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
df.head()
import matplotlib.pyplot as plt
import re
import nltk
X_train=df.comment_text
Y_train=df.drop(['id','comment_text'],axis=1)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
max_words=20000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(X_train))
def preprocess_textdata(text_array,pad_length=200):
    token_txt=tokenizer.texts_to_sequences(text_array)
    padded_txt=pad_sequences(token_txt,maxlen=pad_length)
    return padded_txt
padded_text=preprocess_textdata(X_train,200)
padded_text.shape
from sklearn.model_selection import train_test_split as tts
x_train,x_val,y_train,y_val=tts(padded_text,Y_train,test_size=0.1,random_state=10)
keras.backend.clear_session()
inp=Input(shape=(200,))
embed_size=128
X=Embedding(max_words,embed_size)(inp)
X=Bidirectional(LSTM(60,return_sequences=True))(X)
#X=LSTM(60,return_sequences=True)(X)
X=GlobalMaxPool1D()(X)
X=Dropout(0.1)(X)
X=Dense(60,activation='relu')(X)
X=Dropout(0.1)(X)
X=Dense(6,activation='sigmoid')(X)
model=Model(inputs=inp, outputs=X)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,batch_size=512,epochs=2)
model.evaluate(x_val,y_val,batch_size=512)
inp=Input(shape=(200,))
embed_size=128
X=Embedding(max_words,embed_size)(inp)
#X=Bidirectional(LSTM(60,return_sequences=True))(X)
X=GRU(120,return_sequences=True)(X)
X=GlobalMaxPool1D()(X)
X=Dropout(0.1)(X)
X=Dense(60,activation='relu',kernel_initializer='glorot_uniform')(X)
X=Dropout(0.1)(X)
X=Dense(6,activation='sigmoid',kernel_initializer='glorot_uniform')(X)
model2=Model(inputs=inp, outputs=X)
model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model2.summary()
model2.fit(x_train,y_train,batch_size=512,epochs=3)
model2.evaluate(x_val,y_val,batch_size=512)
testing_df=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
testing_df.head()
x_test=testing_df['comment_text']
padded_test=preprocess_textdata(x_test)
y_test=model2.predict(padded_test,batch_size=512)
testing_df['toxic']=y_test[:,0]
testing_df['severe_toxic']=y_test[:,1]
testing_df['obscene']=y_test[:,2]
testing_df['threat']=y_test[:,3]
testing_df['insult']=y_test[:,4]
testing_df['identity_hate']=y_test[:,5]
testing_df.drop('comment_text',axis=1,inplace=True)
testing_df.head()
testing_df.to_csv('/kaggle/working/out.csv',index=None)