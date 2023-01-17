# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import pickle

# preprocessing
from sklearn import preprocessing

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, GRU, TimeDistributed, Dense, CuDNNGRU, Bidirectional, Dropout
from keras import initializers, regularizers, constraints, optimizers, layers

import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.utils import simple_preprocess
NUM_WORDS = 60000
MAX_NUM_WORDS = NUM_WORDS
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.005
NUM_FILTERS = 25
MAX_LEN = 64 #128 #256
SLICE = 2
Batch_size = 512
train_data = pd.read_csv('../input/train.csv', low_memory=False) # 450MB
test_data = pd.read_csv('../input/test.csv',  low_memory=False) # 20MB
texts = pd.concat([train_data['question_text'], test_data['question_text']]) # 20MB
#texts = pd.concat([train_data['question_text'], test_data['question_text']]) 

# 100MB +-
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(texts)
# transfer sentences into sequences of word indexes
sequences_train = tokenizer.texts_to_sequences(train_data['question_text']) # 300MB +-
sequences_test = tokenizer.texts_to_sequences(test_data['question_text'])
word_index = tokenizer.word_index

print(sequences_train[0])
print('\n')
print('Found %s unique tokens.' % len(word_index))

print(sequences_test[0])
print('\n')
print('Found %s unique tokens.' % len(word_index))
# 1.4GB
X_train = pad_sequences(sequences_train, maxlen=MAX_LEN)
#X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1])
X_test = pad_sequences(sequences_test, maxlen=MAX_LEN)

y_train = train_data['target']

print('Shape of X train: {0}'.format(X_train.shape))
print('Shape of label train: {0}'.format(y_train.shape) )

#load pre-trained GloVe word embeddings
glove_path = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
embeddings_index = {}

f = open(glove_path)
for line in f:
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
#use pre-trained GloVe word embeddings to initialize the embedding layer
embedding_matrix = np.random.random((NUM_WORDS + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    if i<NUM_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be random initialized.
            embedding_matrix[i] = embedding_vector
# 8.4GB
# Load pretrained word vectors
# word_vectors = KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
#                                                  binary=True)

# EMBEDDING_DIM = 300

# # vocab size will be either size of word_index, or Num_words (whichever is smaller)
# vocabulary_size = min(len(word_index) + 1, NUM_WORDS) 

# embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

# # not_found = []
# # found = []

# for word, i in word_index.items():
#     if i>=NUM_WORDS:
#         continue
#     try:
# #         found.append(word)
#         # get vector for each word 
#         embedding_vector = word_vectors[word]
#         # save vector into embedding matrix
#         embedding_matrix[i] = embedding_vector
#     except KeyError:
#         # generate random vector if the word was not found in pretrained vectors
# #         not_found.append(word)
#         embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)
# to free up some memory
#del(word_vectors)
del(sequences_train)
del(sequences_test)
del(tokenizer)
del(texts)
del(train_data)
del(embeddings_index)
# 
#slice sequences into many subsequences
x_test_padded_seqs_split = []

for i in range(X_test.shape[0]):
    split1=np.split(X_test[i], SLICE)
    a=[]
    for j in range(SLICE):
        s=np.split(split1[j], SLICE*4)
        a.append(s)
    x_test_padded_seqs_split.append(a)
x_train_padded_seqs_split = []

for i in range(X_train.shape[0]):
    split1=np.split(X_train[i], SLICE)
    a=[]
    for j in range(SLICE):
        s=np.split(split1[j], SLICE*4)
        a.append(s)
    x_train_padded_seqs_split.append(a)
x_test_padded_seqs_split[1231]
del(X_train)
del(X_test)
embedding_layer = Embedding(MAX_NUM_WORDS + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=4,
                            trainable=True)

#build model
input1 = Input(shape=(4, ), dtype='int32')
embed = embedding_layer(input1)
gru1 = Bidirectional(CuDNNGRU(NUM_FILTERS, return_sequences=False))(embed)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8, 4, ), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = Bidirectional(CuDNNGRU(NUM_FILTERS, return_sequences=False))(embed2)
Encoder2 = Model(input2,gru2)

# expected input_16 to have shape (8, 4, 2) but got array with shape (2, 8, 4)
input3 = Input(shape=(2, 8, 4), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = Bidirectional(CuDNNGRU(NUM_FILTERS, return_sequences=False))(embed3)

preds = Dense(1, activation='sigmoid')(gru3)
model = Model(input3, preds)

print(Encoder1.summary())
print(Encoder2.summary())
print(model.summary())

#use adam optimizer
# from keras.optimizers import Adam
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])


#use the best model to evaluate on test set
# from keras.models import load_model
# best_model= load_model(savebestmodel)          
# print best_model.evaluate(np.array(x_test_padded_seqs_split),y_test,batch_size=Batch_size)
model.fit(np.array(x_train_padded_seqs_split), y_train, validation_split=VALIDATION_SPLIT,
          epochs=2, batch_size=256)
preds = model.predict(np.array(x_test_padded_seqs_split))
#preds = np.round(preds)
preds = (preds > 0.35).astype(np.int)
preds = preds.reshape((preds.shape[0], ))
preds[:3]
df_test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({"qid":df_test.qid, "prediction":preds})
submission.to_csv("submission.csv", index=False)
submission.head(5)
