# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np



text_list  = ['the cat sat on the mat','The dog ate my chocolate']



token_index  = {}



for text in text_list:

    for word in text.split():

        if word not in token_index:

            token_index[word] = len(token_index)+1

max_length = 10



results = np.zeros(shape=(len(text_list),max_length,max(token_index.values())+1))



for i, text in enumerate(text_list):

    print(list(enumerate(text.split())))

    for j,word in list(enumerate(text.split()))[:max_length]:

        index = token_index.get(word)

        results[i,j,index] = 1





print(results)
from keras.preprocessing.text import Tokenizer



text_list  = ['the cat sat on the mat','The dog ate my chocolate']

tokenizer = Tokenizer(num_words= 1000)

# builds word index

tokenizer.fit_on_texts(text_list)



# turns text into list of integer indices

tokenizer.texts_to_sequences(text_list)



one_hot_vectors = tokenizer.texts_to_matrix(text_list,mode = 'binary')

print(one_hot_vectors)

from numpy import array

docs = ['Well done!',

'Good work',

        'Great effort',

        'nice work',

        'Excellent!',

'Weak','Poor effort!',

'not good',

'poor work',

'Could have done better.']

# define class labels

labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents

from keras.preprocessing.text import one_hot

# keras.preprocessing.text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

vocab_size = 50

encoded_docs = [one_hot(d, vocab_size) for d in docs]

print(encoded_docs)
# padding sequence 

from keras.preprocessing.sequence import pad_sequences

# keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)

max_length = 4

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs)

# Defining Embedding layer in keras

from keras.models import Sequential

from keras.layers import Embedding, Flatten,Dense



model = Sequential()

model.add(Embedding(vocab_size, 8, input_length = max_length))

model.add(Flatten())

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# fit the model

model.fit(padded_docs, labels, epochs=50, verbose=0)
