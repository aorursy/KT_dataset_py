# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import re

import operator

from tqdm import tqdm



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score



from keras.models import Sequential, Model

from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout, Input, Embedding, GlobalMaxPool1D

from keras.callbacks import Callback
# Read in data, and take a look

train = pd.read_csv('../input/train.csv')

train, val = train_test_split(train, test_size = 0.2, random_state = 369)

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)
# Initial implementation to user "theoviel"

def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
glove = load_embed ("../input/embeddings/glove.840B.300d/glove.840B.300d.txt")

print("Done extracting GloVe embedding")


def build_vocab(text):

    sentences = text.apply(lambda x: x.split()).values

    vocab = {}

    for s in sentences:

        for w in s:

            try:

                vocab[w] = vocab[w] + 1

            except KeyError:

                vocab[w] = 1

    return vocab



# evaluation of how much coverage there is in the embeddings

def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
# We will apply 2 changes to improve our embedding accuracy:

# 1) Change things to lower case

# 2) Removal special characters and punctuation



punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }



def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown



def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
#train['processed_question'] = train['question_text'].apply(lambda x: x.lower())

#train['processed_question'] = train['processed_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

train['question_text'] = train['question_text'].apply(lambda x: x.lower())

train['question_text'] = train['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
vocab = build_vocab(train['question_text'])

oov_glove = check_coverage(vocab, glove)
# Convert values to embeddings

def text_to_array(text):

    empyt_emb = np.zeros(300)

    text = text[:-1].split()[:30]

    embeds = [glove.get(x, empyt_emb) for x in text]

    embeds+= [empyt_emb] * (30 - len(embeds))

    return np.array(embeds)



# train_vects = [text_to_array(X_text) for X_text in tqdm(train["question_text"])]

val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val["question_text"][:3000])])

val_y = np.array(val["target"][:3000])
# Preprocessing adapted user sudalairajkumar



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# some config values 

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 80 # max number of words in a question to use



# Fill missing values

train_X = train["question_text"].fillna("_na_").values

val_X = val["question_text"].fillna("_na_").values

test_X = test["question_text"].fillna("_na_").values



# Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



# Pad sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



# Get target values

train_y = train['target'].values

val_y = val['target'].values
# Model following: https://www.kaggle.com/nikhilroxtomar/embeddings-cnn-lstm-models-lb-0-683/notebook



all_embs = np.stack(glove.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = glove.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, 300, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNLSTM(64, return_sequences = True))(x)

x = Bidirectional(CuDNNLSTM(64, return_sequences = True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(32, activation = "relu")(x)

x = Dropout(0.3)(x)

#out = Dense(1, activation="sigmoid")(x)

x = Dense(1, activation = "sigmoid")(x)

model = Model(inputs = inp, outputs = x)

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(val_y, (pred_val_y>thresh).astype(int))))
pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = (pred_test_y>0.35).astype(int)

out_df = pd.DataFrame({"qid":test["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)