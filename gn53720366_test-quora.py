from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, CuDNNLSTM
from keras.models import Model

import numpy as np
import pandas as pd
import keras
import os
import re
from tqdm import tqdm
tqdm.pandas()

## 特殊符號
def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

## 二位數～五位數的數字
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

## 一些單字的統稱、過去式跟現在式統一
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'
                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
print(test.shape)
print(train.shape)
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50
y_train = train['target'].values
#train
# train[train['target']==1]
# train.loc[0]['question_text']
# train['question_text'].values
# Clean the text
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: clean_text(x))
    
# Clean numbers
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: clean_numbers(x))
    
# Clean speelings
train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    
# fill up the missing values
# train_X = train["question_text"].fillna("_##_").values
# test_X = test["question_text"].fillna("_##_").values
# load embeddings 
embeddings_glove = {}
with open ('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        line = line.split(" ")
        key = line[0]
        values = np.asarray(line[1:], dtype='float32')
        embeddings_glove[key] = values
#         print(key)
embeddings_glove['a']
# create token 建token表
token = Tokenizer(num_words=MAX_NB_WORDS)
token.fit_on_texts(list(train['question_text'].values))
token.word_index
# 利用建好的token表，將問題字串轉換
x_train = token.texts_to_sequences(train['question_text'].fillna("_##_").values)
x_test = token.texts_to_sequences(test['question_text'].fillna("_##_").values)
print(train['question_text'].values[0])
print(x_train[0])
# 將問題字串截長補短
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
print(x_train[0])
num_words = min(MAX_NB_WORDS, len(token.word_index))
num_words
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# print(embedding_matrix)
embedding_matrix.shape
for word, i in token.word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_glove.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
embedding_matrix
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = Bidirectional(CuDNNGRU(64))(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(x_train, y_train,
          batch_size=128,
          epochs=8, 
          validation_split=0.1)
glove_test_y  = model.predict(x_test, batch_size=128, verbose=1)
# for i in glove_test_y:
#     print(i)
test_y = (glove_test_y>0.4).astype(int)
out = pd.DataFrame({"qid":test["qid"].values})
out['prediction'] = test_y
out.to_csv("submission.csv", index=False)