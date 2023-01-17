# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#%matplotlib inline

import matplotlib.pyplot as plt

from tqdm import tqdm

tqdm.pandas()



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras.optimizers import Adam

from keras import initializers, regularizers, constraints, optimizers, layers





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
# Count rows with missing (or empty string) values

train_df['question_text'].replace('', np.nan, inplace=True)

test_df['question_text'].replace('', np.nan, inplace=True)

print("Train rows with missing values:", train_df.shape[0] - train_df.dropna(subset=['question_text']).shape[0])

print("Test rows with missing values:", test_df.shape[0] - test_df.dropna(subset=['question_text']).shape[0])

train_df.dropna(subset=['question_text'], inplace=True)

test_df.dropna(subset=['question_text'], inplace=True)
# Most (if not all) of the following cleanup code is from https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings



import re



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



def clean_text(x):

    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x



train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





mispell_dict = {'colour':'color', 'centre':'center', 'didnt':'did not', 'doesnt':'does not',

                'isnt':'is not', 'shouldnt':'should not', 'favourite':'favorite',

                'travelling':'traveling', 'counselling':'counseling', 'theatre':'theater',

                'cancelled':'canceled', 'labour':'labor', 'organisation':'organization', 'wwii':'world war 2',

                'citicise':'criticize', 'instagram': 'social medium', 'whatsapp': 'social medium', 'snapchat': 'social medium',

                'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',

                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',

                'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',

                'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018',

                'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp',

                'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'

                }

mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)



train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
train_questions = train_df["question_text"].values

test_questions = test_df["question_text"].values

train_targets = train_df['target'].values
MAX_SEQUENCE_LENGTH = 100

MAX_VOCAB_SIZE = 50000

EMBEDDING_DIM = 300 



## Tokenize the sentences                                                                                                                                                                                   

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(train_questions)

train_questions = tokenizer.texts_to_sequences(train_questions)

test_questions = tokenizer.texts_to_sequences(test_questions)



# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))



# pad sequences so that we get a N x T matrix

train_X = pad_sequences(train_questions, maxlen=MAX_SEQUENCE_LENGTH)

test_X = pad_sequences(test_questions, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of train_X:', train_X.shape)

print('Shape of train_X:', test_X.shape)
# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))
# load in pre-trained Glove word vectors

print('Loading word vectors...')

word2vec = {}

with open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', encoding="utf8") as f:

    for line in f:

        values = line.split(" ")

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32')

        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
# prepare embedding matrix

num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx.items():

    if i < MAX_VOCAB_SIZE:

        embedding_vector = word2vec.get(word)

        if embedding_vector is not None:

        # words not found in embedding index will be all zeros.

            embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an Embedding layer

embedding_layer = Embedding(

  num_words,

  EMBEDDING_DIM,

  weights=[embedding_matrix],

  input_length=MAX_SEQUENCE_LENGTH,

  trainable=False

)
VALIDATION_SPLIT = 0.2

BATCH_SIZE = 512

EPOCHS = 2



input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))

x = embedding_layer(input_)

x = Bidirectional(LSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)





model = Model(input_, x)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])



print(model.summary())
# Train the model

r = model.fit(train_X, train_targets, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
# plot loss and accuracies data

plt.plot(r.history['loss'], label='loss')

plt.plot(r.history['val_loss'], label='val_loss')

plt.legend()

plt.show()



# accuracies

plt.plot(r.history['acc'], label='acc')

plt.plot(r.history['val_acc'], label='val_acc')

plt.legend()

plt.show()
p = model.predict(data)
# Save submission

test_df['prediction'] = (p.flatten() >= 0.5).astype(np.int)

test_df.to_csv('sample_submission.csv', columns=['qid', 'prediction'])