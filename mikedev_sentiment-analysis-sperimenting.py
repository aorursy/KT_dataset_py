# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow_hub as hub

from path import Path

from sklearn.model_selection import train_test_split

import gc

from sklearn.preprocessing import scale, minmax_scale

from sklearn.metrics import log_loss

import hyperopt as hp

import spacy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
IMDB =  Path('../input/aclimdb/aclImdb')

TRAIN = IMDB / 'train'

TEST = IMDB / 'test'

TRAIN_POS = TRAIN / 'pos'

TRAIN_NEG = TRAIN / 'neg'

TEST_POS = TEST / 'pos'

TEST_NEG = TEST / 'neg'
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

embed = hub.Module(module_url)

def load_sentences(is_train=True):

    pos, neg = TRAIN_POS, TRAIN_NEG

    if not is_train:

        pos, neg = TEST_POS, TEST_NEG

    sentences_pos, sentences_neg = [], []

    for fname in pos.listdir('*.txt'):

        with open(fname) as f:

            sentences_pos.append(f.read())

    for fname in neg.listdir('*.txt'):

        with open(fname) as f:

            sentences_neg.append(f.read())

    return sentences_pos, sentences_neg
import spacy

import gc 

nlp = spacy.load('en_core_web_sm')

def spacy_preprocess(texts):

    result = []

    for text in texts:

        doc = nlp(text)

        #advanced model should use also the information about if the word is a noun, verb ecc.

        out = ' '.join([token.lemma_ for token in doc if not token.is_stop])

        result.append(out)

        del doc

        gc.collect()

    return result
def load_and_embed(is_train=True):

    s_pos, s_neg = load_sentences(is_train)

    s_pos = spacy_preprocess(s_pos)

    gc.collect()

    s_neg = spacy_preprocess(s_neg)

    gc.collect()

    embedding_pos = embed(s_pos)

    embedding_neg = embed(s_neg)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    sess.run(tf.tables_initializer())

    X1 = sess.run(embedding_pos)

    X0 = sess.run(embedding_neg)

    sess.close()

    return X1, X0
X1, X0 = load_and_embed(is_train=True)



X = np.concatenate([X1, X0], axis=0)

y = np.concatenate([np.ones([X1.shape[0]]), np.zeros([X0.shape[0]])])

del X1, X0



np.save('X_train.npy', X)

np.save('y_train.npy', y)

del X, y

gc.collect()
X1, X0 = load_and_embed(is_train=False)

X = np.concatenate([X1, X0], axis=0)

y = np.concatenate([np.ones([X1.shape[0]]), np.zeros([X0.shape[0]])])

del X1, X0

np.save('X_test.npy', X)

np.save('y_test.npy', y)

del X, y

gc.collect()