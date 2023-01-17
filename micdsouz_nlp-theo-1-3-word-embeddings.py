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
import gensim 

from gensim.models import KeyedVectors



#loading the downloaded model

EMBEDDING_FILE = '../input/GoogleNews-vectors-negative300.bin'

model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)



#the model is loaded. It can be used to perform all of the tasks mentioned above.
# getting word vectors of a word

dog = model['dog']



#performing king queen magic

print(model.most_similar(positive=['woman', 'king'], negative=['man']))
#picking odd one out

print(model.wv.doesnt_match("breakfast cereal dinner lunch".split()))

#printing similarity index

print(model.similarity('woman', 'man'))
sentence=[["Mary","woman"],["John","is"],["good","boy"]]

#training word2vec on 3 sentences

custommodel = gensim.models.Word2Vec(sentence, min_count=1,size=300,workers=4)

# model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=10, iter=10)
#printing similarity index

print(custommodel.wv.similarity('woman', 'boy'))