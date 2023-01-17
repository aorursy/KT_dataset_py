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
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
print('The shape of train data is:', train_data.shape)
print('The shape of test data is:', test_data.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(ngram_range = (1,3) ,stop_words = 'english')

tfidf_vect.fit_transform(train_data.question_text.tolist() + test_data.question_text.tolist())
train_tfidf = tfidf_vect.transform(train_data.question_text.values.tolist())
test_tfidf = tfidf_vect.transform(test_data.question_text.values.tolist())
train_y = train_data.target.values

#average question text length
ques_text = train_data.question_text.values.tolist() + test_data.question_text.values.tolist()
j = 0
for i in ques_text:
    j += len(i)
print(int(j/len(ques_text)))
import keras
from keras.layers import Dense, Input, Embedding, LSTM, Dropout, Activation
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Bidirectional
test_df = test_data

train_df, val_df = train_test_split(train_data, test_size=0.1, random_state=2018)

embed_dim = 300 #no. of dimensions in word vector
max_features = 50000 #no. of unique words to use
maxlen = 100 #max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

# we now have sequences instead of text. Whatever we use ahead i.e. test_X, train_x, val_X are all sequences



from keras.preprocessing.sequence import pad_sequences
##Padding the sentences(sequence)
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
train_X.shape
## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
from keras.layers import Dense, GlobalMaxPool1D
from keras.models import Model
inp = Input(shape=(maxlen,))
#print(inp.shape)
x = Embedding(max_features,embed_dim)(inp)
#print(x.shape)
x = LSTM(embed_dim, activation='relu')(x)
#print(x.shape)
x = Dense(1, activation='sigmoid', input_shape=(embed_dim,))(x)
#print(x.shape)

model = Model(inputs=inp,outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

model.summary()
model.fit(train_X,train_y, epochs=2, validation_data=(val_X, val_y))