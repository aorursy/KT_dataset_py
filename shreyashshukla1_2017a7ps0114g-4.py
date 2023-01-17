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
data = pd.read_csv('/kaggle/input/nnfl-lab-4/train.csv')

data1 = pd.read_csv('/kaggle/input/nnfl-lab-4/test.csv')

pd.set_option('display.max_colwidth',-1)

data.head()
total_text = pd.concat([data['Sentence1'], data['Sentence2'], data1['Sentence1'], data1['Sentence2']]).reset_index(drop=True)

#total_text1 = pd.concat([data1['Sentence1'], data1['Sentence2']]).reset_index(drop=True)

max_features = 20000

tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',

                                   lower=True,split=' ')

tokenizer.fit_on_texts(total_text)

X1 = tokenizer.texts_to_sequences(data['Sentence1'].values)

X2 = tokenizer.texts_to_sequences(data['Sentence2'].values)

X11 = tokenizer.texts_to_sequences(data1['Sentence1'].values)

X22 = tokenizer.texts_to_sequences(data1['Sentence2'].values)

maxlen = 200

X1 = pad_sequences(X1, maxlen=maxlen)

X2 = pad_sequences(X2, maxlen=maxlen)

X11 = pad_sequences(X11, maxlen=maxlen)

X22 = pad_sequences(X22, maxlen=maxlen)
y= data['Class']

#y11 = data1['Class']

#y=pd.get_dummies(y1)

#y2=pd.get_dummies(y11)

X_train, X_test, y_train, y_test = train_test_split(X1,y, test_size = 0.2, random_state = 42)

X_train1, X_test1, y_train, y_test = train_test_split(X2,y, test_size = 0.2, random_state = 42)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
!wget https://github.com/kmr0877/IMDB-Sentiment-Classification-CBOW-Model/raw/master/glove.6B.50d.txt.gz
! gunzip glove.6B.50d.txt.gz
embeddings_index = {}

f = open('glove.6B.50d.txt')

for line in f:

    values = line.split()

#20/04/2020 Sequence_Models_RNN_Final.ipynb - Colaboratory

#https://colab.research.google.com/drive/1V8gAzzTmiHN1yYG4ya2ptkIXJFK_Mzh3#scrollTo=9ZycSNaqPIyN&printMode=true 10/12

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors in pretrained word vector model.' % len(embeddings_index))

print('Dimensions of the vector space : ', len(embeddings_index['the']))
EMBEDDING_DIM = 50

word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=200,

                            trainable=False)
from keras.layers import Conv1D,MaxPooling1D,Flatten

length_of_text = 200

inputs2 = Input((length_of_text, ))

inputs3 = Input((length_of_text, ))

x2 = embedding_layer(inputs2)

x2 = Conv1D(128, 5, activation='relu')(x2)

x2 = MaxPooling1D(5)(x2)

x2 = Conv1D(128, 5, activation='relu')(x2)

x2 = MaxPooling1D(5)(x2)

x2 = Flatten()(x2)

x3 = embedding_layer(inputs3)

x3 = Conv1D(128, 5, activation='relu')(x3)

x3 = MaxPooling1D(5)(x3)

x3 = Conv1D(128, 5, activation='relu')(x3)

x3 = MaxPooling1D(5)(x3)

x3 = Flatten()(x3)

inputs4 = concatenate([x2,x3],axis=-1)

x4 = Dense(64, activation='relu')(inputs4)

x4 = Dense(1,activation='sigmoid')(x4)

model3 = Model([inputs2,inputs3], x4)

print(model3.summary())
model3.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

model3.fit([X1,X2], y, validation_split=0.2, batch_size = batch_size, epochs = 3)
score,acc = model3.evaluate([X_test,X_test1], y_test, batch_size = batch_size)

print("Validation Accuracy: %.2f" % (acc))
pred = model3.predict([X11,X22])

pred += model3.predict([X22,X11])

pred/=2
data2.head(20)
data2 = pd.read_csv('/kaggle/input/nnfl-lab-4/sample_submission.csv')

#data1['offensive_language'] = data1['offensive_language'].astype(float)

#a=np.zeros((len(pred),2),int)

for i in range(len(pred)):

    #a[i][0]=371+i

    #data1['offensive_language'][i]=np.argmax(pred[i])

    if pred[i]>=0.7:

        data2['Class'][i]=1

    else:

        data2['Class'][i]=0

#df = pd.DataFrame(data=a, columns=["ID", "Class"] )

#data2.to_csv('mycsvfilen.csv',index=False)
data2.to_csv('mycsvfilen.csv',index=False)
model3.save_weights("bestmodellab4.h5")
df=data2
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)