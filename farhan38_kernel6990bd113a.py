# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer,text_to_word_sequence

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from collections import Counter

from sklearn.manifold import TSNE

from gensim.models import word2vec

from nltk import word_tokenize

from nltk.corpus import stopwords
review_data='/kaggle/input/sentiment-dataset-try-1/sentiment_dataset_1_100k.csv'
def getReviewSentimentFromFile(file):

    fr=open(file)

    lines=fr.readlines()

    fr.close()

    reviewsentimentList=[]

    for l in lines:

        x=l.split(',')

        reviewsentimentList.append([str.lstrip(str.rstrip(x[0])),str.lstrip(str.rstrip(x[1]))])

    return reviewsentimentList
rsList=getReviewSentimentFromFile(review_data)
len(rsList[:])
rsDF=pd.DataFrame(rsList,columns=['REVIEW','SENTIMENT'])
rsDF.head(15)
X=rsDF['REVIEW']

y=rsDF['SENTIMENT']

y=to_categorical(num_classes=2,y=y)
np.shape(y)
tok=Tokenizer(lower=True,num_words=10000)

tok.fit_on_texts(X)

seqs=tok.texts_to_sequences(X)

padded_seqs=pad_sequences(seqs,maxlen=100)
def createLSTM():

    model=Sequential()

    model.add(Embedding(10000,100))

    model.add(LSTM(256))

    model.add(Dense(100,activation='sigmoid'))

    model.add(Dense(2,activation='sigmoid'))

    return model
model=createLSTM()

model.summary()
X_train,X_test,y_train,y_test=train_test_split(padded_seqs,y,train_size=0.80,test_size=0.20,random_state=43)
np.shape(X_train),np.shape(y_train)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

history=model.fit(X_train,y_train,batch_size=32,epochs=7,verbose=1)
print(history.history.keys())
plt.plot(history.history['acc'])



plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()
model.evaluate(X_test,y_test)[0]*100
idx=np.random.randint(len(rsDF['REVIEW']))

print(rsDF['REVIEW'].iloc[idx])

test=[rsDF['REVIEW'].iloc[idx]]

test_seq=pad_sequences(tok.texts_to_sequences(test),maxlen=100)

pred=model.predict(test_seq)

proba=model.predict_proba(test_seq)

if np.argmax(pred)==0:

    print('NEG',proba[0][0]*100)

else:

    print('POS',proba[0][1]*100)