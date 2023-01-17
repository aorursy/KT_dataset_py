# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.utils import np_utils

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn import preprocessing

from keras.utils import to_categorical

from keras.layers.recurrent import LSTM

from keras.layers import Dense, GlobalAveragePooling1D, Embedding

from keras.layers import Bidirectional

from keras.layers.normalization import BatchNormalization

from keras import regularizers

from keras.callbacks import EarlyStopping

import time



#read the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

content1=[] #train_content

content2=[] #test content



for i in range(train.shape[0]):

    mytext=train.iloc[i,1]

    content1.append(mytext)

for i in range(test.shape[0]):

    mytext=test.iloc[i,1]

    content2.append(mytext)



#word embedding,got x

tokenizer = Tokenizer(num_words=None)

tokenizer.fit_on_texts(content1 + content2)

train_seq= tokenizer.texts_to_sequences(content1)

test_seq=tokenizer.texts_to_sequences(content2)

train_pad = pad_sequences(train_seq,maxlen=256,padding='post') #pad to the same length

test_pad = pad_sequences(test_seq,maxlen=256,padding='post')

print('train input data')

print(train_pad)
#y one hot
label= preprocessing.LabelEncoder()

label_y= label.fit_transform(train.author.values)

y = to_categorical(label_y,num_classes=3)

print(y)
train_x, test_x, train_y, test_y = train_test_split(train_pad,y,random_state=42,test_size=0.2)

time_start=time.time()
model = Sequential()

model.add(Embedding(len(tokenizer.word_index) + 1, 256))

model.add(Bidirectional(LSTM(256,dropout=0.3,kernel_regularizer=regularizers.l2(0.03),return_sequences=True)))

model.add(Bidirectional(LSTM(256,dropout=0.3,kernel_regularizer=regularizers.l2(0.03))))

model.add(Dropout(0.2))

model.add(Dense(3))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

earlystop = EarlyStopping(monitor='val_loss', patience=2)

hist=model.fit(train_x, train_y,batch_size=16,epochs=25,validation_data=(test_x, test_y),callbacks=[earlystop])
#