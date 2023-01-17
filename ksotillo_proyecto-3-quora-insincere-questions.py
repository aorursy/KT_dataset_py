import pandas as pd

from tqdm import tqdm

tqdm.pandas() # Esto es para poder ver progreso de los datos que se están procesando porque 

              # sino no ando cloro de que está pasando y me pieldo
train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
sentences = train["question_text"].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

print({k: vocab[k] for k in list(vocab)[:5]})
from gensim.models import KeyedVectors



news_path = '../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
oov = check_coverage(vocab,embeddings_index)
oov[:10]
'?' in embeddings_index
'&' in embeddings_index
def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))

sentences = train["question_text"].apply(lambda x: x.split())

vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)
oov[:10]
for i in range(10):

    print(embeddings_index.index2entity[i])
import re



def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))

sentences = train["question_text"].progress_apply(lambda x: x.split())

vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)
oov[:20]
def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





mispell_dict = {'colour':'color',

                'centre':'center',

                'didnt':'did not',

                'doesnt':'does not',

                'isnt':'is not',

                'shouldnt':'should not',

                'favourite':'favorite',

                'travelling':'traveling',

                'counselling':'counseling',

                'theatre':'theater',

                'cancelled':'canceled',

                'labour':'labor',

                'organisation':'organization',

                'wwii':'world war 2',

                'citicise':'criticize',

                'instagram': 'social medium',

                'whatsapp': 'social medium',

                'snapchat': 'social medium'



                }

mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)
train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

sentences = train["question_text"].progress_apply(lambda x: x.split())

to_remove = ['a','to','of','and']

sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]

vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)
oov[:20]
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train, test_size=0.1)
word2vecDict = embeddings_index

embeddings_index = dict()

for word in word2vecDict.wv.vocab:

    embeddings_index[word] = word2vecDict.word_vec(word)

print('Loaded %s word vectors.' % len(embeddings_index))
# Convert values to embeddings

def text_to_array(text):

    empyt_emb = np.zeros(300)

    text = text[:-1].split()[:30]

    embeds = [embeddings_index.get(x, empyt_emb) for x in text]

    embeds+= [empyt_emb] * (30 - len(embeds))

    return np.array(embeds)



# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]

val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])

val_y = np.array(val_df["target"][:3000])

# Data providers

batch_size = 128



def batch_gen(train_df):

    n_batches = math.ceil(len(train_df) / batch_size)

    while True: 

        train_df = train_df.sample(frac=1.)  # Shuffle the data.

        for i in range(n_batches):

            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]

            text_arr = np.array([text_to_array(text) for text in texts])

            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])
from keras.models import Sequential

from keras.layers import CuDNNLSTM, Dense, Bidirectional

import tensorflow as tf
#inp = tf.keras.layers.Input(shape=(10,))

#emb = tf.keras.layers.Embedding(20, 4)(inp)

#x2 = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(emb)

#max_pl = tf.keras.layers.GlobalMaxPooling1D()(x2)

#x = tf.keras.layers.Dense(16, activation="relu")(max_pl)

#x = tf.keras.layers.Dropout(0.1)(x)

#output = tf.keras.layers.Dense(1, activation="sigmoid")(x)



#model = tf.keras.models.Model(inputs=inp, outputs=output)



model = tf.keras.models.Sequential()

x = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64, return_sequences=True),input_shape=(30, 300))

model.add(x)

y = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64))

model.add(y)

model.add(tf.keras.layers.Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()


mg = batch_gen(train_df)







model.fit_generator(mg, epochs=20,

                    steps_per_epoch=1000,

                    validation_data=(val_vects, val_y),

                    verbose=True)
import gc

del val_vects, val_y, train_df, val_df, oov, sentences, to_remove, vocab, mispell_dict, news_path

gc.collect()


# prediction part

batch_size = 256

def batch_gens(test_df):

    n_batches = math.ceil(len(test_df) / batch_size)

    for i in range(n_batches):

        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]

        text_arr = np.array([text_to_array(text) for text in texts])

        yield text_arr





all_preds = []

for x in tqdm(batch_gens(test)):

    all_preds.extend(model.predict(x).flatten())

    gc.collect()
y_te = (np.array(all_preds) > 0.5).astype(np.int)



submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_te})

submit_df.to_csv("submission.csv", index=False)



import matplotlib.pyplot as plt

submit_df['prediction'].plot(kind='hist', bins=100)

plt.xlabel('prediction')