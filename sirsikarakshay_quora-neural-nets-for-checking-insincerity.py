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


#import the required dataset
data=pd.read_csv("../input/train.csv")
data.head()

#select the required data for processing.
req_data=data[['question_text','target']]
req_data.head()
#perform the basic nlp operations on the questions.

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

lines=req_data['question_text'].values.tolist()

#store the english stopwords in a dictionary.
stops=set(stopwords.words('english'))

#create a list for storing the tokens.
quest_tokens=list()

#perform the basic operations required.

for line in lines:
    #convert the words to tokens.
    
    tokens=word_tokenize(line) 
    
    #convert the tokens to lower case.
    
    lower=[w.lower() for w in tokens]
    
    #remove all the punctuations.
    
    no_punct=[w.translate(str.maketrans('','',string.punctuation)) for w in lower]
    
    #remove all the stopwords.
    
    no_stops=[w for w in no_punct if w not in stops]
    
    #remove all non alphabetic characters
    
    final_tokens=[l for l in no_stops if l.isalpha()]
    
    #append the tokens to the list.
    quest_tokens.append(final_tokens)


#build a word2vec model.

import gensim
from gensim.models import Word2Vec

w2v_model=Word2Vec(quest_tokens,size=100,window=5,workers=4,min_count=1)

print(len(w2v_model.wv.vocab))

#Build a tokenizer.

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

#find the maximum length of the questions.
max_len=max(len(s.split())for s in req_data['question_text'])

#initialise a tokenizer object
tokenizer=Tokenizer()

tokenizer.fit_on_texts(quest_tokens)
sequences=tokenizer.texts_to_sequences(quest_tokens)

word_index=tokenizer.word_index

#pad the sequences 
padded_sequences=pad_sequences(sequences,maxlen=max_len)
reviews=req_data['target']

#build the embedding matrix.

num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in word_index.items():
    if i>num_words:
        continue
    embedding_vector=word_index.get(word)
    
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

#Build the keras model.

from keras.models import Sequential
from keras.layers import LSTM,GRU,Dense
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

model=Sequential()

model.add(Embedding(num_words,
                    100,
                    embeddings_initializer=Constant(embedding_matrix),
                   input_length=max_len,
                   trainable=False))

model.add(LSTM(units=32,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(units=8,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(units=1,activation='relu'))

#compile the model.
model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')

#Seperate the training and testing model.

val_split=0.2
indices=np.arange(padded_sequences.shape[0])
np.random.shuffle(indices)

padded_sequences=padded_sequences[indices]
reviews=reviews[indices]

#seperate training and testing.
validation_samples=int(padded_sequences.shape[0]*val_split)

x_train=padded_sequences[:-validation_samples]
y_train=reviews[:-validation_samples]
x_test=padded_sequences[-validation_samples:]
y_test=reviews[-validation_samples:]


#Run the model.
model.fit(x_train,y_train,batch_size=128,epochs=3,validation_data=(x_test,y_test),verbose=2)