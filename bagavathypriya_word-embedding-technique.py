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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
sent =[ 
    'The glass of milk',
    'Glass of juice',
    'I am a good girl',
    'I am a data scientist',
    'Understand the meaning of the words',
    'I like your posts'
]
sent
#Vocabulary size
vocab=10000
onehot_rep = [ one_hot(words,vocab) for words in sent]
print(onehot_rep)
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
#Pad_sequence is used to make all the sentences of same length
sen_length=8 
embedded_sen=pad_sequences(onehot_rep,padding='pre',maxlen=sen_length)
embedded_sen
#Dimension for the embedding matrix
dim=10
model=Sequential()
model.add(Embedding(vocab,dim,input_length=sen_length))
model.compile(optimizer='adam',metrics='mse')
model.predict(embedded_sen)
model.summary()
embedded_sen[0]
model.predict(embedded_sen[0])

