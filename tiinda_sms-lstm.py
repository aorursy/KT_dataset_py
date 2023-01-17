# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_rawSMS = pd.read_csv("../input/spam.csv",encoding='latin-1')
data_rawSMS.head()
def Separate_TrainAndTest(data_rawSMS):
    n=int(data_rawSMS.shape[0])
    tmp_train=(np.random.rand(n)>=0.5)
    return data_rawSMS.iloc[np.where(tmp_train==True)[0]], data_rawSMS.iloc[np.where(tmp_train==False)[0]]
data_rawtrain,data_rawtest=Separate_TrainAndTest(data_rawSMS)
data_rawtrain.head()
data_rawtest.head()
for index, row in data_rawtrain.iterrows():
    if(row['v1'] == 'ham' ):
        data_rawtrain.loc[index, 'new_label'] = 0
    else:
        data_rawtrain.loc[index, 'new_label'] = 1
y_train=data_rawtrain['new_label']
train_text=data_rawtrain['v2']
for index, row in data_rawtest.iterrows():
    if(row['v1'] == 'ham' ):
        data_rawtest.loc[index, 'new_label'] = 0
    else:
        data_rawtest.loc[index, 'new_label'] = 1
y_test=data_rawtest['new_label']
test_text=data_rawtest['v2']
from keras.preprocessing.text import Tokenizer
token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)
from keras.preprocessing import sequence
x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=380)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=3800, 
                    input_length=380))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(units=1, activation='sigmoid' ))

model.compile(loss='binary_crossentropy', 
              #optimizer='rmsprop', 
              optimizer='adam', 
              metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
#filepath="SaveModel/LSTM/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="./SMS_Spam_LSTM_BestWeight.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
train_history =model.fit(x_train, y_train, batch_size=100, 
                         epochs=10, verbose=1,
                         validation_split=0.2, callbacks=callbacks_list)
