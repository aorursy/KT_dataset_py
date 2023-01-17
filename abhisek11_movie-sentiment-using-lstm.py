# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
import os
# print(os.listdir("../input"))

from bs4 import BeautifulSoup

from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import brown
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import gensim

from sklearn.model_selection import train_test_split

model = gensim.models.Word2Vec(brown.sents())
model.save('brown.embedding')
new_model = gensim.models.Word2Vec.load('brown.embedding')
train = pd.read_csv('../input/train.tsv', sep="\t")
train = train[:150000]
test = pd.read_csv('../input/test.tsv', sep="\t")
test = test[:test.shape[0]]
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
# print(train.shape, test.shape, sub.columns)
## Preprocessing and Tokenization

def cleanText(text):
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("  ", " ", text)
    text = " ".join([lemmatiser.lemmatize(w) for w in text.split(' ')])
    return (text.lower())


def data_lstm(df):
    data = list()
    embedding_size = 100
    word_vector_size = 5500
    df['text'] = df['Phrase'].apply(cleanText).apply(word_tokenize)
    max_value = df['text'].apply(len).max()* embedding_size
#     max_value = 48
    test_number = df.shape[0]
    # Using existing word2vec models to convert text to numbers
    for i in range(test_number):
#         print(i)
        val = list()
        for a in range(len(df.loc[i,'text'])):
#             if(len(df.loc[i,'text']) == max_value):
#                 break
                try:
                    val = np.append(val, new_model.wv[df.loc[i,'text'][a]], axis = 0)
                except:
                    pass
#         if(word_vector_size > max_value):
        padding_count = word_vector_size - int(len(val))
        data.append(np.reshape(np.concatenate((np.zeros(padding_count), val), axis=None),(int(word_vector_size/100),embedding_size))) 
#         else:
            
    data_array = np.array(data)
    return(data_array)

# print(type(data), type(data_array), len(data), data_array.shape)
# divide = int(0.8* int(test_number))
data_array = data_lstm(train)

y = pd.get_dummies(train['Sentiment'])
divide =int(0.8*y.shape[0])
y_train = y[:divide]
y_val=y[divide:]
data_array_train = data_array[:divide]
data_array_val = data_array[divide:]
del(train, data_array)
# defining the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(data_array_train.shape[1], data_array_train.shape[2]), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fitting the model
model.fit(data_array_train, y_train, validation_data = (data_array_val, y_val), epochs=4, batch_size=256, verbose = 1)
model.save('trained_model')
del(data_array_train, data_array_val)
# Prediction
x_test = data_lstm(test)
# Y_predict = model.predict(data_array_val)
# scores = model.evaluate(data_array_val, y_val, verbose=1, batch_size=16)
# print(scores)
# print(type(Y_predict), Y_predict, x_test.shape)
#Predict and Submit
Y_predict = model.predict(x_test)
submit = pd.DataFrame()
submit['PhraseId'] = test['PhraseId']
submit['Sentiment'] = np.round(np.argmax(Y_predict, axis=1)).astype(int)
submit.to_csv('Sub_1.csv',index=False)
print(submit.head(5))

