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
import nltk

import pandas as pd

import numpy as n
df=pd.read_csv('../input/imdb-sentiment-analysis/labeledTrainData.tsv',delimiter='\t')

df.head()
df.info()
df['sentiment'].value_counts()
## We will define a function to clean up the message like removing punctuations, single letter words etc..

import re

def clean_str(string):

  """

  String cleaning before vectorization

  """

  try:    

    string = re.sub(r'^https?:\/\/<>.*[\r\n]*', '', string, flags=re.MULTILINE)

    string = re.sub(r"[^A-Za-z]", " ", string)         

    words = string.strip().lower().split()    

    words = [w for w in words if len(w)>=1]

    return " ".join(words)	

  except:

    return ""
df['clean_review']=df['review'].apply(clean_str)
df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    df['review'],

    df['sentiment'],

    test_size=0.2, 

    random_state=42

)
from keras.preprocessing.text import Tokenizer
top_words=10000
t=Tokenizer(top_words)

t.fit_on_texts(X_train)
#t.word_index.items()
X_train=t.texts_to_sequences(X_train)

X_test=t.texts_to_sequences(X_test)
from keras.preprocessing import sequence

max_review_length=300

X_train=sequence.pad_sequences(X_train,maxlen=max_review_length,padding='post')

X_test=sequence.pad_sequences(X_test,maxlen=max_review_length,padding='post')
print(X_train.shape)

print(X_test.shape)
import gensim

word2vec=gensim.models.Word2Vec.load('../input/word2vec-movie-review/word2vec-movie-review_1')

word2vec.wv.syn0.shape
# we will create an embeding Matrix for our Vocab. Above gensim model already has the vector representation of 

# all the majority of the words. We will leverage the model and try to get the vectors of our use case 

# and we will store them in a matrix



embeding_vector_length=word2vec.wv.syn0.shape[1]

embeding_matrix=np.zeros((top_words+1,embeding_vector_length))

for word,i in sorted(t.word_index.items(),key=lambda x:x[1]):

    if i>top_words:

        break

    if word in word2vec.wv.vocab:

        embeding_vector=word2vec.wv[word]

        embeding_matrix[i]=embeding_vector
# Just to have a glance at the vector for our words 

embeding_matrix
# Import the required package from keras library

from keras.models import  Sequential

from keras.layers import Embedding,Dropout,Dense,LSTM
model=Sequential()
model.add(Embedding(top_words+1,50,input_length=max_review_length,weights=[embeding_matrix],trainable=False))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,epochs=10,batch_size=100,validation_data=(X_test,y_test))