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
df=pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
df.head(10)
def classes(category):
        if(category=='ham'):
            return 1
        else:
           return 0
df['Category']=df['Category'].apply(classes)
df.tail(10)
df.isnull().sum()
df.describe()
df.dtypes
messages=np.asarray(df['Message'])
messages
classes=np.asarray(df['Category'])
classes
from keras.preprocessing.text import Tokenizer
final=[]
def total(messages):
    for i in messages:
        final.append(messages)  
    
df['Message'].nunique()
tokenizer = Tokenizer(num_words=10000)#keep 10000 most frequent classes,ignore the others
tokenizer.fit_on_texts(messages)
sequences = tokenizer.texts_to_sequences(messages)
sequences[0]
len(sequences)
len(sequences[0])
len(sequences[100])
max=0
for i in sequences:
    h=len(i)
    if(h>max):
        max=h
print('Maximum sequence length in the list of sequences:', max)
from keras.preprocessing.sequence import pad_sequences
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=189)
# word_index
data
X=data
X.shape
Y=df['Category']
import keras.utils
Y=keras.utils.to_categorical(Y)
Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Embedding,SimpleRNN,LSTM
model = Sequential()
model.add(Embedding(input_dim=10000,output_dim=32,input_length=189))
model.add(SimpleRNN(units=32))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
 metrics=['acc'])
model.summary()
batch_size = 60
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, validation_split=0.2)
acc = model.evaluate(X_test,Y_test)
print("Test loss is {0:.2f} accuracy is {1:.2f} ".format(acc[0],acc[1]))
def message_to_array(msg):
 msg = msg.lower().split(' ')
 test_seq = np.array([word_index[word] for word in msg])
 test_seq = np.pad(test_seq, (189-len(test_seq), 0),
 'constant', constant_values=(0))
 test_seq = test_seq.reshape(1, 189)
 return test_seq
custom_msg = 'Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed and completely free entry for movies'
test_seq = message_to_array(custom_msg)
pred = model.predict_classes(test_seq)
print(pred)
custom_msg = 'Okay fine i will be coming with you we will be having a great time together'
test_seq = message_to_array(custom_msg)
pred = model.predict_classes(test_seq)
print(pred)
model1 = Sequential()
model1.add(Embedding(input_dim=10000,output_dim=32,input_length=189))
model1.add(LSTM(200, dropout_U=0.2,dropout_W=0.2))
model1.add(Dense(2,activation='sigmoid'))
model1.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
model1.summary()
batch_size = 60
model1.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, validation_split=0.2)
acc = model1.evaluate(X_test,Y_test)
print("Test loss is {0:.2f} accuracy is {1:.2f} ".format(acc[0],acc[1]))
custom_msg = 'Congratulations ur awarded 500 of CD vouchers'
test_seq = message_to_array(custom_msg)
pred = model1.predict_classes(test_seq)
print(pred)
custom_msg1 = 'I got another job The one at the hospital doing data analysis or something starts on Monday Not sure when my thesis will finish'
test_seq = message_to_array(custom_msg1)
pred = model1.predict_classes(test_seq)
print(pred)
custom_msg1 = 'If he started searching he will get job in few days He has great potential and talent'
test_seq = message_to_array(custom_msg1)
pred = model1.predict_classes(test_seq)
print(pred)