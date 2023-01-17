# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For examplye, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import, division



import os

import time

import numpy as np

import pandas as pd

import gensim

from tqdm import tqdm

from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()

from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")

import gc



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



import sys

from os.path import dirname

#sys.path.append(dirname(dirname(__file__)))

from keras import initializers

from keras.engine import InputSpec, Layer

from keras import backend as K



import spacy

# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec



# Use fast text as vocabulary

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 

    "Probability of `word`."

    # use inverse of rank as proxy

    # returns 0 if the word isn't in the dictionary

    return - WORDS.get(word, 0)

def correction(word): 

    "Most probable spelling correction for word."

    return max(candidates(word), key=P)

def candidates(word): 

    "Generate possible spelling corrections for word."

    return (known([word]) or known(edits1(word)) or [word])

def known(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)

def edits1(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)

def edits2(word): 

    "All edits that are two edits away from `word`."

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def singlify(word):

    return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])


# modified version of 

# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings

# https://www.kaggle.com/danofer/different-embeddings-with-attention-fork

# https://www.kaggle.com/shujian/different-embeddings-with-attention-fork-fork

def load_glove(word_dict, lemma_dict):

    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    embed_size = 300

    nb_words = len(word_dict)+1

    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)

    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    print(unknown_vector[:5])

    for key in tqdm(word_dict):

        word = key

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.lower()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.upper()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.capitalize()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = ps.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lc.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = sb.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lemma_dict[key]

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        if len(key) > 1:

            word = correction(key)

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:

                embedding_matrix[word_dict[key]] = embedding_vector

                continue

        embedding_matrix[word_dict[key]] = unknown_vector                    

    return embedding_matrix, nb_words 



def load_fasttext(word_dict, lemma_dict):

    EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    embed_size = 300

    nb_words = len(word_dict)+1

    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)

    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    print(unknown_vector[:5])

    for key in tqdm(word_dict):

        word = key

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.lower()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.upper()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.capitalize()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = ps.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lc.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = sb.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lemma_dict[key]

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        if len(key) > 1:

            word = correction(key)

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:

                embedding_matrix[word_dict[key]] = embedding_vector

                continue

        embedding_matrix[word_dict[key]] = unknown_vector                    

    return embedding_matrix, nb_words 





def build_model_simplelstm(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam' ,  metrics=['accuracy'] )



    return model



def build_model2(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.3)(x)

    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)

    x2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x1),

        GlobalMaxPooling1D()(x2),

    ])

    

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'] )



    return model

start_time = time.time()

print("Loading data ...")

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

#train_text = train['comment_text']

#test_text = test['comment_text']

#text_list = pd.concat([train_text, test_text])

#num_train_data = y.shape[0]

print("--- %s seconds ---" % (time.time() - start_time))
#train = train.head(1000000)

#test = test.head(1000000)
gc.collect()


start_time = time.time()

print("Spacy NLP ...")

nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])

nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

word_dict = {}

word_index = 1

lemma_dict = {}

docs = nlp.pipe(pd.concat([train['comment_text'], test['comment_text']]), n_threads = 2)

word_sequences = []

for doc in tqdm(docs):

    word_seq = []

    for token in doc:

        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):

            word_dict[token.text] = word_index

            word_index += 1

            lemma_dict[token.text] = token.lemma_

        if token.pos_ is not "PUNCT":

            word_seq.append(word_dict[token.text])

    word_sequences.append(word_seq)

del docs

gc.collect()

num_train_data = len(train)

train_word_sequences = word_sequences[:num_train_data]

test_word_sequences = word_sequences[num_train_data:]

print("--- %s seconds ---" % (time.time() - start_time))
del nlp



gc.collect()
start_time = time.time()

print("Build spell correction dictnary ...")

spell_model = gensim.models.KeyedVectors.load_word2vec_format('../input/wikinews300d1mvec/wiki-news-300d-1M.vec')

words = spell_model.index2word

w_rank = {}

for i,word in enumerate(words):

    w_rank[word] = i

WORDS = w_rank

print("--- %s seconds ---" % (time.time() - start_time))
del spell_model

del words

del w_rank

gc.collect()
start_time = time.time()

print("Loading embedding matrix ...")

embedding_matrix_glove, nb_words = load_glove(word_dict, lemma_dict)

#embedding_matrix_fasttext, nb_words = load_fasttext(word_dict, lemma_dict)

#embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_fasttext), axis=1)

print("--- %s seconds ---" % (time.time() - start_time))
#del embedding_matrix_glove

#del embedding_matrix_fasttext

#gc.collect()
del word_dict

del lemma_dict

gc.collect()
# hyperparameters

max_length = 220

embedding_size = 300

num_epoch = 4



train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')

test_word_sequences = pad_sequences(test_word_sequences, maxlen=max_length, padding='post')

print(train_word_sequences[:1])

print(test_word_sequences[:1])

pred_prob = np.zeros((len(test_word_sequences),), dtype=np.float32)
gc.collect()
IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train[column] = np.where(train[column] >= 0.5, True, False)

sample_weights = np.ones(len(train), dtype=np.float32)

sample_weights += np.array(train[IDENTITY_COLUMNS].sum(axis=1))

sample_weights += train[TARGET_COLUMN] * (~train[IDENTITY_COLUMNS]).sum(axis=1)

sample_weights += (~train[TARGET_COLUMN]) * train[IDENTITY_COLUMNS].sum(axis=1) * 5

sample_weights /= sample_weights.mean()
y_train = train[TARGET_COLUMN].values

y_aux_train = train[AUX_COLUMNS].values
import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler



NUM_MODELS = 2

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

start_time = time.time()

print("Start training ...")





checkpoint_predictions = []

weights = []



for model_idx in range(1):

    model = build_model_simplelstm(embedding_matrix_glove, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            train_word_sequences,

            [y_train, y_aux_train],

            batch_size=400,

            epochs=1,

            verbose=1,

            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

            callbacks=[

                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict(test_word_sequences, batch_size=2048)[0].flatten())

        result_ = pd.DataFrame.from_dict({

                'id': test.id,

                'prediction': checkpoint_predictions[-1]

                })

        result_.to_csv("model_%d_%d.csv" % (model_idx, global_epoch))

        weights.append(2 ** global_epoch)

   



print("--- %s seconds ---" % (time.time() - start_time))
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)



submission = pd.DataFrame.from_dict({

    'id': test.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)
'''

start_time = time.time()

print("Start training ...")

model = build_model_(embedding_matrix, nb_words, embedding_size)

model.fit(train_word_sequences, y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)

pred_prob += 0.15*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))

model.fit(train_word_sequences, y, batch_size=batch_size, epochs=1, verbose=2)

pred_prob += 0.35*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))

del model, embedding_matrix_fasttext, embedding_matrix

gc.collect()

K.clear_session()

print("--- %s seconds ---" % (time.time() - start_time))

'''










'''

start_time = time.time()

print("Start training ...")

model = build_model(embedding_matrix, nb_words, embedding_size)

model.fit(train_word_sequences, y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)

pred_prob += 0.15*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))

model.fit(train_word_sequences, y, batch_size=batch_size, epochs=1, verbose=2)

pred_prob += 0.35*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))

del model, embedding_matrix_fasttext, embedding_matrix

gc.collect()

K.clear_session()

print("--- %s seconds ---" % (time.time() - start_time))



start_time = time.time()

print("Loading embedding matrix ...")

embedding_matrix_para, nb_words = load_para(word_dict, lemma_dict)

embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_para), axis=1)

print("--- %s seconds ---" % (time.time() - start_time))



start_time = time.time()

print("Start training ...")

model = build_model(embedding_matrix, nb_words, embedding_size)

model.fit(train_word_sequences, y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)

pred_prob += 0.15*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))

model.fit(train_word_sequences, y, batch_size=batch_size, epochs=1, verbose=2)

pred_prob += 0.35*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))

print("--- %s seconds ---" % (time.time() - start_time))



submission = pd.DataFrame.from_dict({'qid': test['qid']})

submission['prediction'] = (pred_prob>0.35).astype(int)

submission.to_csv('submission.csv', index=False)

'''