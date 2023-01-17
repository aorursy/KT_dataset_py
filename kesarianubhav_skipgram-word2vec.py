# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/alicedataset"))



# Any results you write to the current directory are saved as output.
import numpy as np

np.random.seed(13)



from keras.models import Sequential, Model

from keras.layers import Embedding, Reshape, Activation, Input

from keras.layers.merge import Dot

from keras.utils import np_utils

from keras.utils.data_utils import get_file

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import skipgrams

import numpy as np

np.random.seed(13)

from keras.models import Sequential

from keras.layers import Dense, Embedding, Reshape

from IPython.display import SVG

from keras.utils import np_utils

from keras.utils.data_utils import get_file

from keras.preprocessing.text import Tokenizer

# from keras.utils.visualize_util import model_to_dot, plot

# from gensim.models.doc2vec import Word2Vec



import gensim
path = '../input/alicedataset/alice.txt'

corpus = open(path).readlines()

corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

V = len(tokenizer.word_index) + 1

V
# import gensim.downloader as api
# path=api.load("fake-news")
# print(path)

# for i in path:

#     print(i)
dim_embedddings = 128



# inputs

w_inputs = Input(shape=(1, ), dtype='int32')

w = Embedding(V, dim_embedddings)(w_inputs)



# context

c_inputs = Input(shape=(1, ), dtype='int32')

c  = Embedding(V, dim_embedddings)(c_inputs)

o = Dot(axes=2)([w, c])

o = Reshape((1,), input_shape=(1, 1))(o)

o = Activation('sigmoid')(o)



SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)

SkipGram.summary()

SkipGram.compile(loss='binary_crossentropy', optimizer='adam')



for _ in range(5):

    loss = 0.

    for i, doc in enumerate(tokenizer.texts_to_sequences(corpus)):

        data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=5, negative_samples=5.)

        x = [np.array(x) for x in zip(*data)]

        y = np.array(labels, dtype=np.int32)

        if x:

            loss += SkipGram.train_on_batch(x, y)



    print(loss)
d={}

# f = open('vectors.txt' ,'w')

# f.write('{} {}\n'.format(V-1, dim_embedddings))

vectors = SkipGram.get_weights()[0]

for word, i in tokenizer.word_index.items():

    d[word]=list(vectors[i, :])

# f.close()
print(d['city'])
def avg_sentence_vector(words, model, num_features):

    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    for word in words:

        if word in model.keys():

            featureVec = np.add(featureVec, model[word])



    return featureVec
sentence1 = ['i','will','be','eating','tea']

sentence2 = ['i','will','be','drinking','coffee']
feature1 = avg_sentence_vector(sentence1,d,128)

feature2 = avg_sentence_vector(sentence2,d,128)
from scipy.spatial.distance import cosine

print(cosine(feature1,feature2))
sentence3 = ['I','am','walking','to','America']

sentence4 = ['I','am','going','to','Bharat']
feature3 = avg_sentence_vector(sentence3,d,128)

feature4 = avg_sentence_vector(sentence4,d,128)
print(cosine(feature3,feature4))
# import cv2