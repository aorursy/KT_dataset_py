# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from numpy import array

from keras.preprocessing.text import one_hot

from keras.models import Sequential

from keras.layers.embeddings import Embedding

from numpy import asarray

from numpy import zeros

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Embedding


# define documents

docs = ['Well done!',

        'Good work',

        'Great effort',

        'nice work',

        'Excellent!',

        'Weak',

        'Poor effort!',

        'not good',

        'poor work',

        'could have done better.',

        'the King is angry',

        'eldest Son of the king is the right hier',

        'the king of chola fell in love with the queen of chera',

        'the queen looked very beautiful',

        'king is a strong man',

        'kings are men',

        'queens are women',

        'king and queen lived together happily',

        'man are good King',

        'women are good queen as well',

        'queen are good women',

       'the queen was sad waiting for the king to return',

       'king father is dead',

       'king married the queen']







# define class labels

labels = array([1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1])



# Count of unique words in the list of documents

vocab_size = 70





encoded_docs = [one_hot(d, vocab_size) for d in docs]

print(encoded_docs)
# pad documents to a max length of 12 words

# since maximum length in your list of doc is 12

max_length = 12



padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# print(padded_docs)
# define the model

model = Sequential()

model.add(Embedding(vocab_size, 8, input_length=max_length,name='embedding'))



model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='categorical_crossentropy')

# model.summary()

# output_array = model.predict(padded_docs)

# output_array


# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())

# fit the model

model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print('Accuracy: %f' % (accuracy*100))
layer = model.get_layer('embedding' )

output_embeddings = layer.get_weights()
len(output_embeddings[0])
king = output_embeddings[0][13]

queen = output_embeddings[0][30]

woman = output_embeddings[0][40]

man = output_embeddings[0][54]



weak = output_embeddings[0][51]

poor = output_embeddings[0][53]

good = output_embeddings[0][23]


q = (king - man) + woman

q
queen
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(q.reshape(1,8),queen.reshape(1,8))
cosine_similarity(weak.reshape(1,8),good.reshape(1,8))
cosine_similarity(weak.reshape(1,8),poor.reshape(1,8))
# # vectors

# import numpy as np

# a = q

# b = queen

 

# # manually compute cosine similarity

# dot = np.dot(a, b)

# norma = np.linalg.norm(a)

# normb = np.linalg.norm(b)

# cos = dot / (norma * normb)

# cos
from keras.preprocessing.text import Tokenizer



# define documents

docs = ['Well done!',

        'Good work',

        'Great effort',

        'nice work',

        'Excellent!',

        'Weak',

        'Poor effort!',

        'not good',

        'poor work',

        'Could have done better.']

# define class labels

labels = array([1,1,1,1,1,0,0,0,0,0])

# prepare tokenizer

t = Tokenizer()

t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1
# integer encode the documents

encoded_docs = t.texts_to_sequences(docs)

print(encoded_docs)


# pad documents to a max length of 4 words

max_length = 4

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs)


import csv

import os

import pandas as pd



words_vec = pd.read_csv('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
words_vec.shape


# create a weight matrix for words in training docs

embedding_matrix = zeros((vocab_size, 100))

for word, i in t.word_index.items():

    embedding_vector = words_vec[words_vec.index == word].iloc[:,:].values

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
king = words_vec[words_vec.index == 'king'].iloc[:,:].values

queen = words_vec[words_vec.index == 'queen'].iloc[:,:].values

man = words_vec[words_vec.index == 'man'].iloc[:,:].values

woman = words_vec[words_vec.index == 'woman'].iloc[:,:].values

good = words_vec[words_vec.index == 'good'].iloc[:,:].values

bad = words_vec[words_vec.index == 'bad'].iloc[:,:].values
q = (king - man) + woman

cosine_similarity(queen.reshape(1,100),q.reshape(1,100))
# define model

model = Sequential()

e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)

model.add(e)

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))


# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())

# fit the model

model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print('Accuracy: %f' % (accuracy*100))
from sklearn.decomposition import PCA

words = ['king','queen','man','woman','good','bad','cricket','ball','chess','water','rain','island','ocean','red','color','bright']

pca = PCA(n_components=2)

result = pca.fit_transform(words_vec.loc[words])
from matplotlib import pyplot 

# create a scatter plot of the projection

pyplot.scatter(result[:, 0], result[:, 1])



for i, word in enumerate(words):

    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

result = pca.fit_transform(words_vec.loc[words])

from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection='3d')

for index in range(1,len(result)):

    x = result[index][0]

    y = result[index][1]

    z = result[index][2]

    ax.scatter(x, y, z, color='b')

    ax.text(x, y, z, '%s' % (words[index]), size=12, zorder=1, color='k')

pyplot.draw()