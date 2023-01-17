''' This shows an example of converting Newsgroup dataset text into Glove 
Embeddings veectors.

Such embedding is used to convert textual data into numerical form and by using
Glove vectors each word vector preserve its semantic meaning as well.

GloVe embedding data can be found at:
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

BASE_DIR = ''
GLOVE_DIR = r'../input/glove6b100dtxt'
TEXT_DATA_DIR = r'../input/20-newsgroup-original/20_newsgroup/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts.' % len(texts))


labels_index #labels of diiferent text folder
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
# Below line converts all text into sequence of text.
sequences = tokenizer.texts_to_sequences(texts) 

print(sequences[0]) # Token representation of 1st sequence
word_index = tokenizer.word_index
word_index
print("Unique words found {}".format(len(word_index)))
# Creating another data with sequences instead of text.
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# To convert different folder labels into integer values
labels = to_categorical(np.asarray(labels)) 
print("Data shape: ",data.shape)
print("Labels shape: ",labels.shape)
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# Dividing dataset
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1) #20000
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) # (20000,100)

for word, i in word_index.items(): # for every word
    if i >= MAX_NUM_WORDS: # if word_index > 20000
        continue
    embedding_vector = embeddings_index.get(word) # Get vector representation of that word
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Each index of Embed-matrix contain index of that word in word_index and its vectors.
embedding_matrix
# Now form a embedding layer for LSTM model

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(input_dim=num_words, # Size of vocab
                            output_dim=EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH, # 20000
                            trainable=False
                           )
