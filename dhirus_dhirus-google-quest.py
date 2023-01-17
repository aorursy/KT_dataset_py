# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer, one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from keras.callbacks import EarlyStopping

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

train.head()
train.info()
inputs=train['question_title']+' '+ train['question_body']+' '+ train['answer']

inputs
labels = [

        'question_asker_intent_understanding',

        'question_body_critical',

        'question_conversational',

        'question_expect_short_answer',

        'question_fact_seeking',

        'question_has_commonly_accepted_answer',

        'question_interestingness_others',

        'question_interestingness_self',

        'question_multi_intent',

        'question_not_really_a_question',

        'question_opinion_seeking',

        'question_type_choice',

        'question_type_compare',

        'question_type_consequence',

        'question_type_definition',

        'question_type_entity',

        'question_type_instructions',

        'question_type_procedure',

        'question_type_reason_explanation',

        'question_type_spelling',

        'question_well_written',

        'answer_helpful',

        'answer_level_of_information',

        'answer_plausible',

        'answer_relevance',

        'answer_satisfaction',

        'answer_type_instructions',

        'answer_type_procedure',

        'answer_type_reason_explanation',

        'answer_well_written'    

    ]
target = train[labels].values
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS =40000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH =150

# This is fixed.

EMBEDDING_DIM = 300

def get_coefs(word,*arr): 

    return word, np.asarray(arr, dtype='float32')
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS =40000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH =150

# This is fixed.

EMBEDDING_DIM = 50



tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(inputs)
## Using Glove 50d embedding



EMBEDDING_FILE = '../input/glove50d/glove.6B.50d.txt'

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

emb_mean,emb_std
word_index = tokenizer.word_index

nb_words = min(MAX_NB_WORDS, len(word_index))
word_index = tokenizer.word_index

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))

for word, i in word_index.items():

    if i >= MAX_NB_WORDS: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
inp = Input(shape=(MAX_SEQUENCE_LENGTH,))

x = Embedding(MAX_NB_WORDS,EMBEDDING_DIM, weights=[embedding_matrix])(inp)

x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(30, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
X = tokenizer.texts_to_sequences(inputs.values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
epochs =50

batch_size =1000

model.fit(X,target, epochs=epochs, batch_size=batch_size,validation_split=0.1)