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
import pandas as pd

import numpy as np

import re

from tqdm import tqdm



from nltk.corpus import stopwords



from gensim.models import Word2Vec

from gensim.models import Phrases

from gensim.models.phrases import Phraser



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from keras.utils import np_utils

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Embedding, Dropout,Bidirectional, Reshape, Flatten, CuDNNGRU, CuDNNLSTM

from keras.models import Model, Sequential

from keras.initializers import Constant

from keras import backend as K
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
# 불용어 만들기

stopWords = stopwords.words('english')

# 데이터 정제

def cleanData(sentence):

    processedList = ""

    

    # convert to lowercase, ignore all special characters - keep only alpha-numericals and spaces (not removing full-stop here)

    sentence = re.sub(r'[^A-Za-z0-9\s.]',r'',str(sentence).lower())

    sentence = re.sub(r'\n',r' ',sentence)

    

    # remove stop words

    sentence = " ".join([word for word in sentence.split() if word not in stopWords])

    

    return sentence





train_df['question_text'] = train_df['question_text'].apply(lambda x :cleanData(x))

test_df['question_text'] = test_df['question_text'].apply(lambda x :cleanData(x))

train_X = train_df['question_text']

test_X = test_df['question_text']

print(train_X.shape)

print(test_X.shape)
# corpus 만들기

tmp_corpus = train_X.apply(lambda x: x.split("."))

corpus = []

for i in tqdm(range(len(tmp_corpus))):

    for line in tmp_corpus[i]:

        words = [x for x in line.split()]

        corpus.append(words)
#keras로 전처리

maxlen = 70 

tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(train_X))



train_X = tokenizer.texts_to_sequences(train_X)

test_X = tokenizer.texts_to_sequences(test_X)



train_X = pad_sequences(train_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



train_y = train_df['target'].values

train_y = np_utils.to_categorical(train_y)



print(train_X.shape)

print(test_X.shape)

print(train_y.shape)
#gensim으로 Word2Vec 만들기

model = Word2Vec(corpus, sg =1, window = 3, size = 100, min_count = 5, workers = 4 , iter = 100)

filename = 'gensim_word2vec.txt'

model.wv.save_word2vec_format(filename, binary = False)
#Word2Vec 사용하기

import os

embedding_index = {}

f = open(os.path.join("",'gensim_word2vec.txt'), encoding = 'utf-8')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:])

    embedding_index[word] = coefs

f.close()



word_index = tokenizer.word_index



num_words = len(word_index) + 1

embedding_matrix = np.zeros((num_words, 100))



for word, i in word_index.items():

    if i > num_words:

        continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
#model 만들기

model1 = Sequential()

model1.add(Embedding(num_words, 100, embeddings_initializer= Constant(embedding_matrix),trainable=False, input_length = 70))

model1.add(Bidirectional(CuDNNLSTM(70, return_sequences=True)))

model1.add(Bidirectional(CuDNNLSTM(70, return_sequences=True)))

model1.add(Bidirectional(CuDNNLSTM(70, return_sequences=True)))

model1.add(Flatten())

model1.add(Dense(100, activation = 'relu'))

model1.add(Dense(100, activation = 'relu'))

model1.add(Dense(2, activation = 'sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#model1.summary()

model1.fit(train_X, train_y, batch_size = 500, epochs = 7)
pred_y = np.argmax(model1.predict(test_X), axis = 1)

test_df = pd.read_csv("../input/test.csv", usecols=["qid"])

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_y

out_df.to_csv("submission.csv", index=False)