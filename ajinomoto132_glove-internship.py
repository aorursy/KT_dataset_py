import numpy as np

import pandas as pd

import os

import time

import gc

import random

import spacy

import pickle

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from keras.preprocessing import text, sequence
from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()

from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")



def is_interactive():

    return 'SHLVL' not in os.environ



if not is_interactive():

    def nop(it, *a, **k):

        return it
true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')

true['label'] = 0



cleansed_data = []

for data in true.text:

    if "@realDonaldTrump : - " in data:

        cleansed_data.append(data.split("@realDonaldTrump : - ")[1])

    elif "(Reuters) -" in data:

        cleansed_data.append(data.split("(Reuters) - ")[1])

    else:

        cleansed_data.append(data)



true["text"] = cleansed_data

true.head(5)
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')

fake['label'] = 1



dataset = pd.concat([true, fake])

dataset = dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)

dataset = dataset.iloc[:7500]

dataset.head()
GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

PARAM_PATH = '../input/pickled-param/pickled-param.pickle'

MAX_LEN = 200
start_time = time.time()

text_list = dataset.text

print("Spacy NLP ...")

nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])

nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

word_dict = {}

word_index = 1

lemma_dict = {}

docs = nlp.pipe(text_list, n_threads = 2)

word_sequences = []

for doc in tqdm(docs):

    word_seq = []

    for token in doc:

        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):

            word_dict[token.text] = word_index

            word_index += 1

            lemma_dict[token.text] = token.lemma_

        if token.pos_ is not "PUNCT":

            word_seq.append(token.text)

    word_sequences.append(word_seq)

del docs

gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    if '.pkl' in path or '.pickle' in path:

        with open(path,'rb') as f:

            return pickle.load(f)

    else:

        with open(path, encoding="utf8", errors='ignore') as f:

            return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))



    

def P_glove(word): 

    "Probability of `word`."

    # use inverse of rank as proxy

    # returns 0 if the word isn't in the dictionary

    return - word_dict.get(word, 0)

def correction_glove(word): 

    "Most probable spelling correction for word."

    return max(candidates_glove(word), key=P_glove)

def candidates_glove(word): 

    "Generate possible spelling corrections for word."

    return (known_glove([word]) or known_glove(edits1_glove(word)) or [word])

def known_glove(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in word_dict)

def edits1_glove(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)
embeddings_index = load_embeddings(GLOVE_EMBEDDING_PATH)
x_train = []

for sequence in word_sequences:  

    if len(sequence) > MAX_LEN:

        sequence = sequence[:MAX_LEN]

    else:

        padding = MAX_LEN - len(sequence)

        sequence.extend(['unknown']*padding)

    matrix = []

    for word in sequence:

        try:

            seq = embeddings_index[word]

        except:

            seq = embeddings_index['unknown']

        matrix.append(list(seq))

    x_train.append(np.array(matrix))

    

del word_sequences, sequence, matrix, seq, embeddings_index, word_dict

gc.collect()
np.array(x_train).shape
import tensorflow as tf



# this is the size of our encoded representations

encoding_dim = 150



# this is our input placeholder

input_ = tf.keras.layers.Input(shape=(None,300))

# "encoded" is the encoded representation of the input

encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_)



# "decoded" is the lossy reconstruction of the input

decoded = tf.keras.layers.Dense(300, activation='sigmoid')(encoded)



# this model maps an input to its reconstruction

autoencoder = tf.keras.models.Model(input_, decoded)



# intermediate result

# this model maps an input to its encoded representation

encoder = tf.keras.models.Model(input_, encoded)



autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
x_train = np.array(x_train)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5, 

                                                          verbose = 1, min_delta = 0.0001, restore_best_weights = True)



autoencoder.fit(x_train, x_train,

                epochs=30,

                batch_size=128,

                shuffle=True,

                )
# reconst_test = autoencoder.predict(X_train)

encode_test = encoder.predict(x_train)

encode_test = encode_test.reshape(dataset.shape[0],200*encoding_dim)
from sklearn import cluster



# Training for 2 clusters (Fake and Real)

kmeans = cluster.KMeans(n_clusters=2, verbose=1)



# Fit predict will return labels

clustered = kmeans.fit_predict(encode_test)
correct = 0

incorrect = 0

for index, row in enumerate(dataset['label'].values):

    if row == clustered[index]:

        correct += 1

    else:

        incorrect += 1

        

print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
# maxlen = 512

# X = []

# for text in dataset.text[:100]:

#     enc_di = tokenizer.encode(text).ids

#     len_ = len(enc_di)

    

#     if len_ > maxlen:

#         enc_di = enc_di[:500] + enc_di[len_-(maxlen-500):]

#     else:

#         padding_len = maxlen-len_

#         enc_di += [1] * (padding_len)

    

#     input_ids = tf.constant(enc_di)[None, :]  # Batch size 1

#     outputs = model(input_ids)

#     last_hidden_states = outputs[0].numpy().squeeze(0)

#     X.append(last_hidden_states)
# np.array(X).shape