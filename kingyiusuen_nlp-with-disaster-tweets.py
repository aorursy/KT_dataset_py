import numpy as np

import pandas as pd

from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()

from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")

import gc

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPooling1D, Concatenate, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Sequential

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import Constant

import spacy

import pickle

import re

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train.head()
train['keyword'].value_counts().sort_values(ascending=False)
pd.crosstab(train['keyword'].isnull(), train['target'])
train['location'].value_counts().sort_values(ascending=False)[0:20]
train['location'].isnull().sum() / train.shape[0]
pd.crosstab(train['location'].isnull(), train['target'])
train['target'].value_counts()
def count_regex(pattern, tweet):

    return len(re.findall(pattern, tweet))

    

for df in [train, test]:

    df['words_count'] = df['text'].apply(lambda x: count_regex(r'\w+', x))

    df['unique_words_count'] = df['text'].apply(lambda x: len(set(str(x).split())))

    df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    df['chars_count'] = df['text'].apply(lambda x: len(str(x)))

    df['mentions_count'] = df['text'].apply(lambda x: count_regex(r'@\w+', x))

    df['hashtags_count'] = df['text'].apply(lambda x: count_regex(r'#\w+', x))

    df['capital_words_count'] = df['text'].apply(lambda x: count_regex(r'\b[A-Z]{2,}\b', x))

    df['excl_quest_marks_count'] = df['text'].apply(lambda x: count_regex(r'!|\?', x))

    df['urls_count'] = df['text'].apply(lambda x: count_regex(r'http.?://[^\s]+[\s]?', x))
new_features = ['words_count', 'unique_words_count', 'mean_word_length', 'chars_count', 'mentions_count', 

                'hashtags_count', 'capital_words_count', 'excl_quest_marks_count', 'urls_count']

disaster_tweets_idx = train['target'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(new_features), figsize=(20, 50), dpi=100)



for i, feature in enumerate(new_features):

    sns.distplot(train.loc[~disaster_tweets_idx][feature], label='Not Disaster', ax=axes[i][0], color='green')

    sns.distplot(train.loc[disaster_tweets_idx][feature], label='Disaster', ax=axes[i][0], color='red')



    sns.distplot(train[feature], label='Train', ax=axes[i][1])

    sns.distplot(test[feature], label='Test', ax=axes[i][1])

    

    for j in range(2):

        axes[i][j].set_xlabel('')

        axes[i][j].tick_params(axis='x', labelsize=12)

        axes[i][j].tick_params(axis='y', labelsize=12)

        axes[i][j].legend()

    

    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)

    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)



plt.show()
train_text = train['text']

test_text = test['text']

target = train['target'].values



# combine the text from the training dataset and the test dataset

text_list = pd.concat([train_text, test_text])



# number of training samples

num_train_data = target.shape[0]
text_list.iloc[171]
text_list = text_list.apply(lambda x: re.sub('&amp;', ' and ', x))

text_list = text_list.apply(lambda x: re.sub('w/', 'with', x))
# https://www.kaggle.com/wowfattie/3rd-place

nlp = spacy.load('en_core_web_lg')

nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

docs = nlp.pipe(text_list, n_threads = 2)



# convert words to integers and save the results in word_sequences

word_sequences = []



# store the mapping in word_dict

word_dict = {}

lemma_dict = {}



# store the frequence of each word

word_freq = {}



word_index = 1

for doc in docs:

    word_seq = []

    for word in doc:

        try:

            word_freq[word.text] += 1

        except KeyError:

            word_freq[word.text] = 1

        if (word.text not in word_dict) and (word.pos_ is not 'PUNCT'):

            word_dict[word.text] = word_index

            word_index += 1

            lemma_dict[word.text] = word.lemma_

        # do not include punctuations in word_dict

        # this essentially removes hashtags and mentions

        if word.pos_ is not 'PUNCT':

            word_seq.append(word_dict[word.text])

    word_sequences.append(word_seq)

del docs

gc.collect()



# maximum number of words per tweet in the dataset

max_length = max([len(s) for s in word_sequences])



# number of unique words

# add 1 because 0 is reserved for padding

vocab_size = len(word_dict) + 1



train_word_sequences = word_sequences[:num_train_data]

test_word_sequences = word_sequences[num_train_data:]



# add zeros at the end of each word sequence so that their lengths are fixed at max_len

train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')

test_word_sequences = pad_sequences(test_word_sequences, maxlen=max_length, padding='post')
def load_embeddings(embeddings_index, word_dict, lemma_dict):

    embed_size = 300

    vocab_size = len(word_dict) + 1

    embedding_matrix = np.zeros((vocab_size, embed_size), dtype=np.float32)

    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    for key in word_dict:

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

        embedding_matrix[word_dict[key]] = unknown_vector                    

    return embedding_matrix
def load_pickle_file(path):

    with open(path, 'rb') as f:

        file = pickle.load(f)

    return file



# the asterisk below allows the function to accept an arbitrary number of inputs

def get_coefs(word,*arr): 

    """ convert the embedding file into a Python dictionary """ 

    return word, np.asarray(arr, dtype='float32')

                                                  

path_glove = '/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

path_paragram = '/kaggle/input/paragram-300-sl999/paragram_300_sl999.txt'



embeddings_index_glove = load_pickle_file(path_glove)

# the asterisks below unpacks the list

embeddings_index_paragram = dict(get_coefs(*o.split(" ")) for o in open(path_paragram, encoding="utf8", errors='ignore') if len(o) > 100)



embedding_matrix_glove = load_embeddings(embeddings_index_glove, word_dict, lemma_dict)

embedding_matrix_paragram = load_embeddings(embeddings_index_paragram, word_dict, lemma_dict)



# stack the two pre-trained embedding matrices

embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_paragram), axis=1)
embedding_size = 600

learning_rate = 0.001

batch_size = 32

num_epoch = 5



def build_model(embedding_matrix, vocab_size, max_length, embedding_size=300):

    model = Sequential([

        Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), 

                  input_length=max_length, trainable=False),

        SpatialDropout1D(0.3),

        Bidirectional(LSTM(128, return_sequences=True)),

        Conv1D(64, kernel_size=2), 

        GlobalMaxPooling1D(),

        Dense(1, activation='sigmoid')

    ])

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
reps = 5

pred_prob = np.zeros((len(test_word_sequences),), dtype=np.float32)

for r in range(reps):

    model = build_model(embedding_matrix, vocab_size, max_length, embedding_size)

    model.fit(train_word_sequences, target, batch_size=batch_size, epochs=num_epoch, verbose=2)

    pred_prob += np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2) / reps)
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission['target'] = pred_prob.round().astype(int)

submission.to_csv('submission.csv', index=False)