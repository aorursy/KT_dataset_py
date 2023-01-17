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
import numpy as np

docs = "Can I eat the Pizza".lower().split()

doc1 = set(docs)

doc1 = sorted(doc1)

print ("\nvalues: ", doc1)



integer_encoded = []

for i in docs:

    v = np.where( np.array(doc1) == i)[0][0]

    integer_encoded.append(v)

print ("\ninteger encoded: ",integer_encoded)



def get_vec(len_doc,word):

    empty_vector = [0] * len_doc

    vect = 0

    find = np.where( np.array(doc1) == word)[0][0]

    empty_vector[find] = 1

    return empty_vector



def get_matrix(doc1):

    mat = []

    len_doc = len(doc1)

    for i in docs:

        vec = get_vec(len_doc,i)

        mat.append(vec)

        

    return np.asarray(mat)



print ("\nMATRIX:")

print (get_matrix(doc1))
from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

# define example

# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']





doc1 = "Can I eat the Pizza".lower()

doc2 = "You can eat the Pizza".lower()

doc1 = doc1.split()

doc2 = doc2.split()

doc1_array = array(doc1)

doc2_array = array(doc2)

doc3 = doc1+doc2

# doc3 = set(doc3)

data = list(doc3)





values = array(data)

print(values)

# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)

print(integer_encoded)



# binary encode

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)





# invert first example

inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

print(inverted)
from keras.preprocessing.text import Tokenizer

from numpy import array

from numpy import argmax

from keras.utils import to_categorical





doc = "Can I eat the Pizza".lower().split()



def using_Tokenizer(doc):

    # create the tokenizer

    t = Tokenizer()

    # fit the tokenizer on the documents

    t.fit_on_texts(doc)



    # integer encode documents

    encoded_docs = t.texts_to_matrix(doc, mode='count')

    return encoded_docs



def using_to_categorical(doc):

    label_encoder = LabelEncoder()

    data = label_encoder.fit_transform(doc)

    data = array(data)



    # one hot encode

    encoded = to_categorical(data)

    return encoded



def invert_encoding(row_num):

    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[row_num, :])])

    return inverted

    

print ("===using Keras Tokenizer for OneHotEncoding===")

print (using_Tokenizer(doc))

print ()

print ("===using Keras to_categorical for OneHotEncoding===")

print (using_to_categorical(doc))

print ()

print (invert_encoding(int(0)))
import tensorflow as tf

import pandas as pd



text = 'My cat is a great cat'

tokens = text.lower().split()



vocab = set(tokens)

vocab = pd.Series(range(len(vocab)), index=vocab)



word_ids = vocab.loc[tokens].values



inputs = tf.placeholder(tf.int32, [None])

# TensorFlow has an operation for one-hot encoding

one_hot_inputs = tf.one_hot(inputs, len(vocab))

transformed = tf.Session().run(one_hot_inputs, {inputs: word_ids})





print (transformed)