import numpy as np

import pandas as pd

import os

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model

from keras.layers import Dense, Embedding, LSTM, Input

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re
data = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/nlp_train.csv')

data1 = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')

pd.set_option('display.max_colwidth',-1)

data.head()
num_words = 20000

tokenizer = Tokenizer(num_words=num_words)#, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890',lower=True,split=' ')

#text=pd.concat([data['tweet'],data1['tweet']])

print(data['tweet'][0])

tokenizer.fit_on_texts(data['tweet'].values)

X = tokenizer.texts_to_sequences(data['tweet'].values)

#tokenizer.fit_on_texts(data1['tweet'].values)

X1 = tokenizer.texts_to_sequences(data1['tweet'].values)

print(X[0])

max=0

j=0

for i in range(len(X)):

    if len(X[i])>max:

        max=len(X[i])

        j=i

print(max)

print(j)

print(X[j])

print(data['tweet'][j])
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

max_length_of_text = 200

X = pad_sequences(X, maxlen=max_length_of_text)

X1 = pad_sequences(X1, maxlen=max_length_of_text)

print(word_index)

print("Padded Sequences: ")

print(X)
y = data['offensive_language']

y11 = data1['offensive_language']

#y=pd.get_dummies(y1)

y2=pd.get_dummies(y11)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
from keras.layers import Dropout,GlobalMaxPool1D,Flatten

embed_dim = 128 #Change to observe effects

lstm_out = 60 #Change to observe effects

batch_size = 32

inputs = Input((max_length_of_text, ))

x = Embedding(num_words, embed_dim)(inputs)    # More info on Embeddings here - https://mac

# And here : https://www.kaggle.com/rajmehra03/a-detailed-explanation-of-keras-embedding-lay

#x = LSTM(lstm_out)(x) # The LSTM transforms the vector sequence into a single vector of siz

#x = Conv1D(64, 3, activation='relu')(x)

#x = MaxPooling1D(3)(x)

#x = Conv1D(64, 3, activation='relu')(x)

#x = MaxPooling1D(3)(x)

#x = Flatten()(x)

x = LSTM(lstm_out,return_sequences=True)(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50,activation='relu')(x)

x = Dropout(0.1)(x)

#x = Dense(16,activation='relu')(x)

#x = Dense(32,activation='relu')(x)

x = Dense(1,activation='linear')(x)

model = Model(inputs, x)

print(model.summary())
model.compile(loss = 'mean_absolute_error', optimizer='adam',metrics = ['mse'])

model.fit(X, y, validation_split=0.2, batch_size = batch_size, epochs = 5)
from sklearn.metrics import mean_squared_error

ya=model.predict(X_test)

print(mean_squared_error(ya,y_test)) 
pred = model.predict(X1)

for i in range (20):

    #print(np.argmax(pred[i]))

    print(pred[i])
data1.head()
data1['offensive_language'] = data1['offensive_language'].astype(float)

#a=np.zeros((len(pred),2),int)

for i in range(len(pred)):

    #a[i][0]=371+i

    #data1['offensive_language'][i]=np.argmax(pred[i])

    #if

    data1['offensive_language'][i]=pred[i]

#df = pd.DataFrame(data=a, columns=["ID", "Class"] )

#data1.to_csv('mycsvfile.csv',index=False)
data1.to_csv('mycsvfile.csv',index=False)
model.save_weights("bestmodellab3.h5")
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = data1.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(data1)