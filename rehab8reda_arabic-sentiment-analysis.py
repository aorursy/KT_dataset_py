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
### load data 
neg_train=pd.read_csv('/kaggle/input/arabic-sentiment-twitter-corpus/train_Arabic_tweets_negative_20190413.tsv',sep='\t')
neg_train.columns=['sentiment','text']
len(neg_train)
pos_train=pd.read_csv('/kaggle/input/arabic-sentiment-twitter-corpus/train_Arabic_tweets_positive_20190413.tsv',sep='\t')
pos_train.columns=['sentiment','text']
print(len(pos_train))
print(len(neg_train))
## concatenate data 
data=pd.concat([neg_train,pos_train])
len(data)
## shufle data
from sklearn.utils import shuffle
shuffled_data = shuffle(data)
shuffled_data.head()
## convert pos to 1 and neg to 0
shuffled_data['sentiment']=shuffled_data['sentiment'].apply(lambda x: 1 if x =='pos' else 0)
shuffled_data.head()
## import libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from keras.models import Model ,Sequential
from keras.layers import Embedding ,Dense ,Dropout , LSTM ,Input ,Activation , Bidirectional ,GlobalMaxPool1D
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import glorot_uniform
%matplotlib inline
from gensim.models import KeyedVectors
def load_w2v(file_path,binary):
    return KeyedVectors.load_word2vec_format(file_path,binary=binary)
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ar.vec
w2v = load_w2v("wiki.ar.vec", binary=False)
print(len(w2v.vocab))
max_sequence_length=85
max_nm_words=len(w2v.vocab)
embeding_dim=300 
sample=w2v['حسن']
print(sample.shape)
print(w2v.most_similar('حسن'))
## get stop words
!wget  https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt  -o stop_words.txt
import os 
import glob
import codecs
from keras.preprocessing.text import Tokenizer ,text_to_word_sequence
def get_stop_words():
    path='list.txt'
    stop_words = []
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as myfile:
        stop_words = myfile.readlines()
    stop_words = [word.strip() for word in stop_words]
    return stop_words
stop_words=get_stop_words()
print(stop_words)
## remove stop words from text
shuffled_data['text']=shuffled_data['text'].apply(lambda x : [item for item in x.split() if item not in stop_words])
shuffled_data.text
## tokenize data
tokenizer= Tokenizer()
tokenizer.fit_on_texts(shuffled_data['text'])
word_index=tokenizer.word_index
vocab_size=len(word_index)
print(vocab_size)
train_sequences=tokenizer.texts_to_sequences(shuffled_data['text'])
train_paded_sequences=pad_sequences(train_sequences,maxlen=max_sequence_length,padding='post',truncating='post')
len(train_paded_sequences)
from sklearn.model_selection import train_test_split
train_paded_sequences,valid_paded_sequences,y_train,y_valid=train_test_split(train_paded_sequences,shuffled_data['sentiment'].values,test_size=.2)
embedding_matrix=np.zeros((max_nm_words,embeding_dim))
for word ,i in word_index.items():
    if word in w2v.vocab:
        embedding_matrix[i] = w2v.word_vec(word)
### build model
sentence_indices = Input(shape=(max_sequence_length,),dtype='int32')
embedding_layer = Embedding(vocab_size+1 , embeding_dim,  input_length=max_sequence_length)
embeddings = embedding_layer(sentence_indices)   
X = LSTM(60, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(embeddings)
X = GlobalMaxPool1D()(X)
X = Dropout(0.2)(X)
X = Dense(128)(X)
X = Activation("relu")(X)
X = Dropout(0.2)(X)
X = Dense(512)(X)
X = Activation("relu")(X)
X = Dropout(0.2)(X)
X = Dense(1)(X)
X = Activation('softmax')(X)
model = Model(inputs=sentence_indices,outputs=X)
model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(train_paded_sequences, y_train, batch_size=32, epochs=10, validation_data=(valid_paded_sequences, y_valid))
