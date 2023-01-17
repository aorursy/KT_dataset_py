# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
'''
keras for deep learning models

Preprocessing Imports
'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
'''
Different Neural Network Layers
'''
from keras.layers import Dense,  Input, GlobalMaxPooling1D
from keras.layers import CuDNNGRU, MaxPool1D, Embedding, Bidirectional
from keras.layers import Dropout, SpatialDropout1D
'''
Build Model
'''
from keras.models import Model
# ROC Curve
from sklearn.metrics import roc_auc_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#set configurations and dimensions 

MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 20000

VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300
BATCH_SIZE = 1000
EPOCHS = 10
# Path to data 
train_data_path = '../input/jigsaw-toxic-comment-classification-challenge/train.csv'
test_data_path = '../input/jigsaw-toxic-comment-classification-challenge/test.csv'

# path to GloVe
glove_path = '../input/glove6b/glove.6B.{0}d.txt'.format(EMBEDDING_DIM)

'''
loading word2vectors from GloVe
'''
print ('loading word2vec...')

word2vec = {}

with open(os.path.join(glove_path), encoding='utf8') as fs:
    for line in fs:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print ('number of vectors : {0}'.format(len(word2vec)))
'''
loading training data
'''
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
#loading all row wise comment_text data into sentences
sentences = train_data['comment_text'].fillna('DUMMY_VALUES').values
# storing labels intp possible_labels
possible_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
# loading all row wise possible_labels into target
targets = train_data[possible_labels].values
'''
    converting sentences into interger sequences

'''
# initialize tokenizer
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
# downsizing or fitting the sentences into respective tokens
tokenizer.fit_on_texts(sentences)
# transforming text to integer sequences 
sequences = tokenizer.texts_to_sequences(sentences)
    

# map word to integer [indexing]
word_index = tokenizer.word_index
# number of unique words
print(len(word_index))
# type 
print(type(word_index))
# convert all different input sizes into constant size of max_sequence_length
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
# checking shape of data
print('shape of data {0}'.format(data.shape))
# preparing embedding matrix 
print('Filling pre-trained embeddings...')

num_words = min(MAX_VOCAB_SIZE,len(word_index)+1)

# initially populate embedding matrix to be all zeros
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        
        if embedding_vector is not None:
            # words which are found will be updated
            embedding_matrix[i] = embedding_vector

#shape of embedding_matrix
print('shape of embedding matrix is {0}'.format(embedding_matrix.shape))
# creating a embeddings object for neural net using pretrained weights
embedding_layer = Embedding(
num_words,
EMBEDDING_DIM,
weights =[embedding_matrix],
input_length = MAX_SEQUENCE_LENGTH,
trainable = False
)
print('Building the Model...')
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))

x = embedding_layer(input_)

x = Bidirectional(CuDNNGRU(50, return_sequences= True))(x)

x = SpatialDropout1D(0.1)(x)

x = GlobalMaxPooling1D()(x)

x = Dense(128, activation ='relu')(x)

x = Dropout(0.2)(x)

output = Dense(len(possible_labels), activation = 'sigmoid')(x)

model = Model(input_, output)
model.compile( 
loss = 'binary_crossentropy',
optimizer = 'adam',
metrics = ['accuracy'])
print('Training Model...')
r = model.fit(
    data,
    targets,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split = VALIDATION_SPLIT
)
test_sentences = test_data['comment_text'].fillna('DUMMY_VALUES').values
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_feed = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
predict = model.predict(test_feed)
submission_path = '../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv'

submission  = pd.read_csv(submission_path)

submission[possible_labels] = predict

submission.to_csv('submission.csv', index=False)
