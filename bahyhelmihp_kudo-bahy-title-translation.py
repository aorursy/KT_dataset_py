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
dev_en = pd.read_csv("/kaggle/input/shopee-product-title-translation-open/dev_en.csv")
dev_en.head()
dev_tcn = pd.read_csv("/kaggle/input/shopee-product-title-translation-open/dev_tcn.csv")
dev_tcn.head()
import string
import re
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
import jieba
en_tcn = pd.concat([dev_en['translation_output'], dev_tcn['text']], axis=1).values
en_tcn
# Remove punctuation, symbols, any non words
en_tcn[:,0] = [re.sub(r'[^\w]', ' ', s) for s in en_tcn[:,0]]
en_tcn[:,1] = [re.sub(r'[^\w]', ' ', s) for s in en_tcn[:,1]]
en_tcn
# convert text to lowercase
for i in range(len(en_tcn)):
    en_tcn[i,0] = en_tcn[i,0].lower()
    en_tcn[i,1] = en_tcn[i,1].lower()
en_tcn
## Cut with jieba before tokenize with keras tokenizer, remove extra spaces
en_tcn[:,0] = [re.sub(' +', ' ', s) for s in en_tcn[:,0]]
en_tcn[:,1] = [re.sub(' +', ' ', " ".join(jieba.cut(s, cut_all=False))) for s in en_tcn[:,1]]
en_tcn
# empty lists
eng_l = []
tcn_l = []

# populate the lists with sentence lengths
for i in en_tcn[:,0]:
      eng_l.append(len(i.split()))

for i in en_tcn[:,1]:
      tcn_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'tcn':tcn_l})

length_df.hist(bins = 30)
plt.show()
max(eng_l), max(tcn_l)
# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
# prepare english tokenizer
eng_tokenizer = tokenization(en_tcn[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = max(eng_l)
print('English Vocabulary Size: %d' % eng_vocab_size)
# prepare Chinese tokenizer
tcn_tokenizer = tokenization(en_tcn[:, 1])
tcn_vocab_size = len(tcn_tokenizer.word_index) + 1

tcn_length = max(tcn_l)
print('Chineese Vocabulary Size: %d' % tcn_vocab_size)
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq
from sklearn.model_selection import train_test_split

# split data into train and test set
train, test = train_test_split(en_tcn, test_size=0.2, random_state = 12)
# prepare training data
trainX = encode_sequences(tcn_tokenizer, tcn_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

# prepare validation data
testX = encode_sequences(tcn_tokenizer, tcn_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
# build NMT model
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model
# model compilation
model = define_model(tcn_vocab_size, eng_vocab_size, tcn_length, eng_length, 512)
rms = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')
filename = 'model.h1.31_jul_20'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# train model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=30, batch_size=512, validation_split = 0.2,callbacks=[checkpoint], 
                    verbose=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()
model = load_model('model.h1.31_jul_20')
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                 temp.append('')
            else:
                 temp.append(t)
        else:
            if(t == None):
                  temp.append('')
            else:
                  temp.append(t) 

    preds_text.append(' '.join(temp))
pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
pred_df.sample(10)
# product_title_translation_eval_script.py
"""Sample evaluation script for product title translation."""
from typing import List
import regex
# !pip install sacrebleu
from sacrebleu import corpus_bleu

OTHERS_PATTERN: re.Pattern = regex.compile(r'\p{So}')


def eval(preds: List[str], refs: List[str]) -> float:
    """BLEU score computation.

    Strips all characters belonging to the unicode category "So".
    Tokenize with standard WMT "13a" tokenizer.
    Compute 4-BLEU.

    Args:
        preds (List[str]): List of translated texts.
        refs (List[str]): List of target reference texts.
    """
    preds = [OTHERS_PATTERN.sub(' ', text) for text in preds]
    refs = [OTHERS_PATTERN.sub(' ', text) for text in refs]
    return corpus_bleu(
        preds, [refs],
        lowercase=True,
        tokenize='13a',
        use_effective_order= False
    ).score
eval(pred_df['actual'], pred_df['predicted'])
test_data = pd.read_csv("../input/shopee-product-title-translation-open/test_tcn.csv")
test_data.head()
test_data.shape
def process_test_data(arr):
    ## Remove punctuation, symbols, extra spaces
    arr = arr.apply(lambda x: re.sub(r'[^\w]', ' ', x))
    arr = arr.apply(lambda x: re.sub(' +', ' ', " ".join(jieba.cut(x, cut_all=False))))
    ## To lowercase
    arr = arr.apply(lambda x: x.lower())
    ## Get max length
    tcn_l = []
    for i in arr:
        tcn_l.append(len(i.split()))
    tcn_length = max(tcn_l)
    
    return arr, tcn_length
tcn_arr, tcn_length_test = process_test_data(test_data['text'])
print(tcn_arr.head())
print(tcn_length_test)
# prepare test data
testX_test = encode_sequences(tcn_tokenizer, tcn_length_test, tcn_arr)
preds = model.predict_classes(testX_test.reshape((testX_test.shape[0],testX_test.shape[1])))
preds
preds.shape
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                 temp.append('')
            else:
                 temp.append(t)
        else:
            if(t == None):
                  temp.append('')
            else:
                  temp.append(t) 

    preds_text.append(' '.join(temp))
res = pd.DataFrame({"translation_output": preds_text})
res.shape
str_zero = res['translation_output'].value_counts().index[0]
str_zero
str_most = 'baby'
res['translation_output'] = res['translation_output'].apply(lambda x: "baby" if x == str_zero else x)
res['translation_output'].value_counts()
res.to_csv("submission_title_translation.csv", index=False)
pd.read_csv("./submission_title_translation.csv").shape