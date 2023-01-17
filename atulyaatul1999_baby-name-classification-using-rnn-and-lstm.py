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


df=pd.read_csv('../input/us-baby-names/NationalNames.csv')
df.head()
df.shape
df['Name'].nunique()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


df['Gender']=le.fit_transform(df['Gender'])
df.head()
df.tail()
data=df.groupby('Name').mean()[['Gender']].reset_index()
data.head()
data["Gender"]=data['Gender'].astype('int')
data.head()
data.shape
import string
lower=list(string.ascii_lowercase)
lower
vocab=dict(zip(lower,range(1,27)))
vocab
rev_vocab=dict(zip(range(1,27),lower))
rev_vocab
name="Atul".lower()
name
seq=[vocab[i] for i in name]
seq
X = []
for name in data['Name'].values:
    name = name.lower()
    seq = [vocab[i] for i in name]
    X.append(seq)
X[1]
y=data["Gender"].values
from keras.preprocessing.sequence import pad_sequences
X=pad_sequences(X,maxlen=10,padding='pre')
X[1]
print(X.shape,y.shape)
from keras.layers import *
from keras.models import Model
inp=Input(shape=(10,))
emb=Embedding(input_dim=27,output_dim=5)(inp)
rnn=SimpleRNN(units=32)(emb)
out=Dense(units=1,activation='sigmoid')(rnn)
model=Model(inputs=inp,outputs=out)
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X,y,epochs=10,batch_size=1024,validation_split=0.2)
name="Atul".lower()
seq=[vocab[i] for i in name]
x_test=pad_sequences([seq],maxlen=10,padding='pre')
x_test
model.predict(x_test)
inp=Input(shape=(10,))
emb=Embedding(input_dim=27,output_dim=5)(inp)
lstm=LSTM(units=32)(emb)
out=Dense(units=1,activation='sigmoid')(lstm)
lstm_model=Model(inputs=inp,outputs=out)
lstm_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
lstm_model.fit(X,y,batch_size=1024,epochs=10,validation_split=0.2)
