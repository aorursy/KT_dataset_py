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







""" Load keras library """

from keras.models import Model

from keras.layers.recurrent import LSTM

from keras.layers import Dense, Input, Embedding

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint

from collections import Counter

import nltk

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import casual_tokenize

import re

import random

import time

import pickle





# keras imports, because there are like... A million of them.

from keras.models import Model

from keras.optimizers import Adam

from keras.layers import Dense, Input, LSTM, Dropout, Embedding, RepeatVector, concatenate, TimeDistributed

from keras.utils import np_utils



# 8192 - large enough for demonstration, larger values make network training slower

MAX_VOCAB_SIZE = 2**13

# seq2seq generally relies on fixed length message vectors - longer messages provide more info

# but result in slower training and larger networks

MAX_MESSAGE_LEN = 30  

# Embedding size for words - gives a trade off between expressivity of words and network size

EMBEDDING_SIZE = 100

# Embedding size for whole messages, same trade off as word embeddings

CONTEXT_SIZE = 50

# Larger batch sizes generally reach the average response faster, but small batch sizes are

# required for the model to learn nuanced responses.  Also, GPU memory limits max batch size.

BATCH_SIZE = 64

# Helps regularize network and prevent overfitting.

DROPOUT = 0.2

# High learning rate helps model reach average response faster, but can make it hard to 

# converge on nuanced responses

LEARNING_RATE=0.015



# Tokens needed for seq2seq

UNK = 0  # words that aren't found in the vocab

PAD = 1  # after message has finished, this fills all remaining vector positions

START = 2  # provided to the model at position 0 for every response predicted



# Implementaiton detail for allowing this to be run in Kaggle's notebook hardware

SUB_BATCH_SIZE = 32
DATA_PATH = 'amazon.csv'

WEIGHT_FILE_PATH = 'Model_weights.h5'



# Tokens needed for seq2seq

UNK = 0  # words that aren't found in the vocab

PAD = 1  # after message has finished, this fills all remaining vector positions

START = 2  # provided to the model at position 0 for every response predicted







# Replace anonymized screen names with common token @__sn__

def sn_replace(match):

    _sn = match.group(2).lower()

    if not _sn.isnumeric():

        # This is a company screen name

        return match.group(1) + match.group(2)

    return '@__sn__'



#df=pd.read_csv("D:\\Deep learning\\Project\\Data\\amazon.csv")

#df=df.head(50)

#df.to_csv("D:\\Deep learning\\Project\\Data\\amazon1.csv",index=False,sep = '|')



df=pd.read_csv("../input/amazon.csv",error_bad_lines=True)

df=df.head(500)

sn_re = re.compile('(\W@|^@)([a-zA-Z0-9_]+)')

print("Replacing anonymized screen names in X...")

x_text = df.text_x.apply(lambda txt: sn_re.sub(sn_replace, str(txt)))

print("Replacing anonymized screen names in Y...")

y_text = df.text_y.apply(lambda txt: sn_re.sub(sn_replace, str(txt)))

count_vec = CountVectorizer(tokenizer=casual_tokenize, max_features=MAX_VOCAB_SIZE - 3)

print("Fitting CountVectorizer on X and Y text data...")

count_vec.fit(x_text + y_text)

analyzer = count_vec.build_analyzer()

pickle.dump(count_vec, open("analyzer.pickle", 'wb'),protocol=2)

vocab = {k: v + 3 for k, v in count_vec.vocabulary_.items()}

vocab['__unk__'] = UNK

vocab['__pad__'] = PAD

vocab['__start__'] = START

# Used to turn seq2seq predictions into human readable strings

reverse_vocab = {v: k for k, v in vocab.items()}

print(f"Learned vocab of {len(vocab)} items.")



word2idx = dict([(idx, word) for word, idx in vocab.items()])

idx2word = dict([(idx, word) for word, idx in reverse_vocab.items()])







#print(word2idx)

np.save('word-input-word2idx.npy', word2idx)

np.save('word-input-idx2word.npy', idx2word)





def to_word_idx(sentence):

    full_length = [vocab.get(tok, UNK) for tok in analyzer(sentence)] + [PAD] * MAX_MESSAGE_LEN

    return full_length[:MAX_MESSAGE_LEN]



def from_word_idx(word_idxs):

    return ' '.join(reverse_vocab[idx] for idx in word_idxs if idx != PAD).strip()







print("Calculating word indexes for X...")

x = pd.np.vstack(x_text.apply(to_word_idx).values)

print("Calculating word indexes for Y...")

y = pd.np.vstack(y_text.apply(to_word_idx).values)





all_idx = list(range(len(x)))

train_idx = set(random.sample(all_idx, int(0.8 * len(all_idx))))

test_idx = {idx for idx in all_idx if idx not in train_idx}



train_x = x[list(train_idx)]

test_x = x[list(test_idx)]

train_y = y[list(train_idx)]

test_y = y[list(test_idx)]



assert train_x.shape == train_y.shape

assert test_x.shape == test_y.shape



print(f'Training data of shape {train_x.shape} and test data of shape {test_x.shape}.')

def create_model():

    shared_embedding = Embedding(

        output_dim=EMBEDDING_SIZE,

        input_dim=MAX_VOCAB_SIZE,

        input_length=MAX_MESSAGE_LEN,

        name='embedding',

    )

    

    # ENCODER

    

    encoder_input = Input(

        shape=(MAX_MESSAGE_LEN,),

        dtype='int32',

        name='encoder_input',

    )

    

    embedded_input = shared_embedding(encoder_input)

    

    # No return_sequences - since the encoder here only produces a single value for the

    # input sequence provided.

    encoder_rnn = LSTM(

        CONTEXT_SIZE,

        name='encoder',

        dropout=DROPOUT

    )

    

    #context = RepeatVector(MAX_MESSAGE_LEN)(encoder_rnn(embedded_input))

    

    # DECODER

    

    last_word_input = Input(

        shape=(MAX_MESSAGE_LEN, ),

        dtype='int32',

        name='last_word_input',

    )

    

    embedded_last_word = shared_embedding(last_word_input)

    # Combines the context produced by the encoder and the last word uttered as inputs

    # to the decoder.

    decoder_input = embedded_last_word

    

    # return_sequences causes LSTM to produce one output per timestep instead of one at the

    # end of the intput, which is important for sequence producing models.

    decoder_rnn = LSTM(

        CONTEXT_SIZE,

        name='decoder',

        return_sequences=True,

        dropout=DROPOUT

    )

    

    decoder_output = decoder_rnn(decoder_input)

    

    # TimeDistributed allows the dense layer to be applied to each decoder output per timestep

    next_word_dense = TimeDistributed(

        Dense(int(MAX_VOCAB_SIZE / 2), activation='relu'),

        name='next_word_dense',

    )(decoder_output)

    

    next_word = TimeDistributed(

        Dense(MAX_VOCAB_SIZE, activation='softmax'),

        name='next_word_softmax'

    )(next_word_dense)

    

    return Model(inputs=[encoder_input, last_word_input], outputs=[next_word])



s2s_model = create_model()

optimizer = Adam(lr=LEARNING_RATE, clipvalue=5.0)

s2s_model.compile(optimizer='adam', loss='categorical_crossentropy')



json = s2s_model.to_json()

open('model-architecture.json', 'w').write(json)

def add_start_token(y_array):

    """ Adds the start token to vectors.  Used for training data. """

    return np.hstack([

        START * np.ones((len(y_array), 1)),

        y_array[:, :-1],

    ])



def binarize_labels(labels):

    """ Helper function that turns integer word indexes into sparse binary matrices for 

        the expected model output.

    """

    return np.array([np_utils.to_categorical(row, num_classes=MAX_VOCAB_SIZE)

                     for row in labels])





def respond_to(model, text):

    """ Helper function that takes a text input and provides a text output. """

    input_y = add_start_token(PAD * np.ones((1, MAX_MESSAGE_LEN)))

    idxs = np.array(to_word_idx(text)).reshape((1, MAX_MESSAGE_LEN))

    for position in range(MAX_MESSAGE_LEN - 1):

        prediction = model.predict([idxs, input_y]).argmax(axis=2)[0]

        input_y[:,position + 1] = prediction[position]

    return from_word_idx(model.predict([idxs, input_y]).argmax(axis=2)[0])







def train_mini_epoch(model, start_idx, end_idx):

    """ Batching seems necessary in Kaggle Jupyter Notebook environments, since

        `model.fit` seems to freeze on larger batches (somewhere 1k-10k).

    """

    b_train_y = binarize_labels(train_y[start_idx:end_idx])

    #print(b_train_y)

    input_train_y = add_start_token(train_y[start_idx:end_idx])

    checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

    

    model.fit(

        [train_x[start_idx:end_idx], input_train_y], 

        b_train_y,

        epochs=1,

        batch_size=BATCH_SIZE,callbacks=[checkpoint]

    )

    

    model.save_weights(WEIGHT_FILE_PATH)

    model.save('Model_weights.hdf5')



    rand_idx = random.sample(list(range(len(test_x))), SUB_BATCH_SIZE)

    print('Test results:', model.evaluate(

        [test_x[rand_idx], add_start_token(test_y[rand_idx])],

        binarize_labels(test_y[rand_idx])

    ))

    

    input_strings = [

        "@AmazonHelp Your delivery person left my parcel on my doormat overnight",

 

    ]

    

    for input_string in input_strings:

        output_string = respond_to(model, input_string)

        print(output_string)

        print(f'> "{input_string}"\n< "{output_string}"')













for epoch in range(50):

        print(f'Training in epoch {epoch}...')

        for start_idx in range(0, len(train_x), SUB_BATCH_SIZE):

            train_mini_epoch(s2s_model, start_idx, start_idx + SUB_BATCH_SIZE)
