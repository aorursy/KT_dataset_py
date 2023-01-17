# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pickle

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.models import Model

CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'

GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
def get_coefs(word,*arr):

    return word,np.asarray(arr , dtype = 'float32')
def load_embeddings(path):

    with open(path , 'rb') as f:

        emb_arr = pickle.load(f)

        

        return emb_arr        
def build_matrix(word_index,path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1,300))

    unknown_words = []

    

    for word , i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

            

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix
os.listdir('../input/nlp-getting-started/')
sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub.head()
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
TEXT_COLUMN = 'text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'