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
import seaborn as sns

import tensorflow.keras as keras

from keras.layers import LSTM,Dense,Flatten,Conv1D,Embedding,Dropout

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import re

from keras.optimizers import RMSprop
def stop_words(sentence):

    sentence=sentence.lower()

    sentence=re.sub(r"[^A-Za-z]",' ',sentence)

    s=sentence.split(" ")

    stop=stopwords.words('english')

    h=[]

    for x in s:

        if x not in stop:

            h.append(x)

    return ' '.join(h)
df=pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

df['Category']=df['Category'].replace({'ham':0,'spam':1})

df['removed_stop_words']=df['Message'].apply(lambda x: stop_words(x))

labels=df['Category']

text=df['removed_stop_words']
train_text=text[:4000]

train_labels=labels[:4000]

test_text=text[4000:]

test_labels=labels[4000:]
t=Tokenizer(oov_token='<OOV>')

t.fit_on_texts(train_text)

word_index=t.word_index
tokenized_train_text=t.texts_to_sequences(train_text)

tokenized_test_text=t.texts_to_sequences(test_text)
sentence1=tokenized_train_text[0]

sentence2=tokenized_test_text[1]

index_to_word_dict=dict(zip(word_index.values(),word_index.keys()))
print(word_index['<OOV>'])

print(index_to_word_dict[1])
def convert_seq_to_sentence(seq):

    h=[]

    for x in seq:

        h.append(index_to_word_dict[x])

    return ' '.join(h)
print(convert_seq_to_sentence(sentence1))

print(df['Message'][0])
print(convert_seq_to_sentence(sentence2))

print(df['Message'][4001])
len=70

pad='post'

paded_train_text=pad_sequences(tokenized_train_text,maxlen=len,padding=pad)

paded_text_text=pad_sequences(tokenized_test_text,maxlen=len,padding=pad)
paded_train_text.shape
model=Sequential([

    Embedding(6437,32,input_length=len),

    LSTM(32,return_sequences=True,recurrent_dropout=0.5),

    LSTM(32),

    Flatten(),

    Dropout(0.2),

    Dense(128,activation='relu'),

    Dropout(0.2),

    Dense(128,activation='relu'),

    Dense(1,activation='sigmoid')

])
model.summary()
model.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics='acc')

epoch=15

history=model.fit(paded_train_text,train_labels,epochs=epoch,validation_split=0.2)
d=history.history

plt.plot(range(1,epoch+1),d['acc'],label='train')

plt.plot(range(1,epoch+1),d['val_acc'],label='validation')

plt.legend()
plt.plot(range(1,epoch+1),d['loss'],label='train')

plt.plot(range(1,epoch+1),d['val_loss'],label='validation')

plt.legend()
model.evaluate(paded_text_text,test_labels)
pd.value_counts(df['Category'])
pd.value_counts(np.array(test_labels))