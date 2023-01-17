# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/glovembedding/"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import time



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.models import Model

from keras.layers import Dense, Embedding, Input

from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,CuDNNLSTM, GlobalAveragePooling1D,Concatenate

from keras.optimizers import Adam,RMSprop
data = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

data.head()
train = data[['target','comment_text']]

train.head()
train['target'] = train['target'].apply(lambda x: 1 if x > 0.5 else 0)

train.head()
y_train = train['target']

y_train.head()
def clean(text):

    text = text.str.lower()

    text = text.str.replace(r'\r',' ')

    text = text.str.replace(r'\n',' ')

    text = text.str.replace('[^a-zA-Z0-9]',' ')

    text = text.apply(lambda x: " ".join(x.split()))

    return text
X = data['comment_text']

X = clean(X)

X.head()
le = LabelEncoder()

y_train = le.fit_transform(y_train)

y_train
#Variable Declaration



MAX_SEQUENCE_LENGTH = 200

MAX_VOCAB_SIZE = 50000

EMBEDDING_DIM = 50

BATCH_SIZE = 512

LSTM_UNITS = 128

LSTM_UNITS_2 = 128

EPOCHS = 4
# Tokenizer

stime = time.time()



sentences = X.values

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)



etime = time.time()

# Words are converted into integers for the model



print(sentences[0])

print(sequences[0])



t = etime-stime



print('Execution Time: ',t)
max(len(s.split()) for s in sentences) # As we see that the maximum length is 327 but we are limiting it to 200
datafinal = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', datafinal.shape)
# Get the word to index mapping

word2idx = tokenizer.word_index
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

num_words
len(word2idx) # There are 305408 unique words from the dataset but we are limiting the number of words 20000.
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) # Creating the embedding matrix for 50000 words , each having a EMBEDDING_DIM of 50

embedding_matrix.shape
# Creating the Word2Vec vector from predefined glove word vectors



stime = time.time()



word2vec = {}

with open(os.path.join('../input/glovembedding/glove.6B.50d.txt')) as f:

    for line in f:

        values = line.split()

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32')

        word2vec[word] = vec



etime = time.time()

t = etime-stime

print('Found %s word vectors.' % len(word2vec))

print('Execution time: ',t)



#Creating the embedding vector for all our words 



stime = time.time()



for word, i in word2idx.items():

    if i < MAX_VOCAB_SIZE:

        embedding_vector = word2vec.get(word)

        if embedding_vector is not None:

            # words not found in embedding index will be all zeros.

            embedding_matrix[i] = embedding_vector



etime = time.time()

t = etime-stime

print('Execution time: ',t)
# Adding an embedding layer to the model



embedding_layer = Embedding(

  num_words,

  EMBEDDING_DIM,

  weights=[embedding_matrix],

  input_length=MAX_SEQUENCE_LENGTH,

  trainable=False                   # as the embeddings are pretrained

)

print('Shape of the input: ',datafinal.shape)  # Batch size = 1804874 and Sequence Length = 200
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))  ## First Layer of the network

print('Shape :',input_.shape)     ## It expects every row to have 200 columns or 200 words in our case
## Adding the embedding layer to our input



x = embedding_layer(input_)                 ## Second Layer of the network

print('Shape :',x.shape)   ## Now we see that the size has become 200*50 as each of the word is now represented by 50 vectors
## Adding a Bidirectional LSTM layer to the embedding output



x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x) ## Third Layer of the network

x = Dropout(0.2)(x)

print('Shape :',x.shape)

x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x) ## Third Layer of the network

x = Dropout(0.2)(x)

print('Shape :',x.shape)                     ## Now we see a size of 200 as the the 50 size vector is now represented by 200 hidden states of the LSTM considering Bidirectional
## Adding a MaxPooling Layer



x1 = GlobalMaxPool1D()(x)         ## Fourth Layer of the network

x2 = GlobalAveragePooling1D()(x)

concatenate = Concatenate()

x = concatenate([x1,x2])

print('Shape :',x.shape)         ## It has performed a maximum function on axis 1 and now we have 2 dim instead of 3 as required by the Dense Layer to follow
## Adding Dense Layers



# x = Dense(1024,activation = 'relu')(x)  # Fifth Layer

# x = Dropout(0.2)(x)

# print('Shape :',x.shape)            # Now the size has changed form 200 to 512

x = Dense(512,activation = 'relu')(x)  # Sixth Layer

x = Dropout(0.2)(x)

print('Shape :',x.shape)           

x = Dense(64,activation = 'relu')(x)   # Seventh Layer

x = Dropout(0.2)(x)

print('Shape :',x.shape)            

output = Dense(1, activation="sigmoid")(x)  # Output Layer

print('Shape :',output.shape)            # Final output

model = Model(inputs = input_, outputs = output)  # Initialting the model



model.compile(

  loss='binary_crossentropy',     ## Assiging Loss 

  optimizer=Adam(lr=0.01,amsgrad = True),        ## Optimizer with Learning Rate

  metrics=['accuracy']            ## Metric 

)
#y[0:5]
r = model.fit(

  datafinal,

  y_train,

  batch_size=BATCH_SIZE,

  epochs=EPOCHS,

  validation_split = 0.1

)
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
test_data = clean(test['comment_text'])
test_data = tokenizer.texts_to_sequences(test_data)

test_data = pad_sequences(test_data, maxlen=MAX_SEQUENCE_LENGTH)

test_data[0:1]
test_data.shape
stime = time.time()



y = model.predict(test_data)



etime = time.time()



print(etime-stime)

#y = np.where(y>=0.5,1,0)

#y = y.astype(np.int32)

y
sub_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

sub_df['prediction'] = y
sub_df.to_csv('submission.csv', index=False)