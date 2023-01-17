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
import pickle



def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr
from keras.preprocessing import text



tokenizer = text.Tokenizer(num_words=100, filters='',lower=False)

tokenizer.fit_on_texts(['привет', 'как дела', 'любой другой текст'])
MAX_FEATURES = 30
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((MAX_FEATURES + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        if i <= MAX_FEATURES:

            try:

                embedding_matrix[i] = embedding_index[word]

            except KeyError:

                try:

                    embedding_matrix[i] = embedding_index[word.lower()]

                except KeyError:

                    try:

                        embedding_matrix[i] = embedding_index[word.title()]

                    except KeyError:

                        unknown_words.append(word)

    return embedding_matrix, unknown_words
emb_matrix, unknown_words = build_matrix(tokenizer.word_index, '../input/cc.ru.300.pickle')