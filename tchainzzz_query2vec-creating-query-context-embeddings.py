# data processing

import pandas as pd

import numpy as np

import nltk

import itertools

import random

import hashlib

from keras.preprocessing.text import hashing_trick

from keras.utils import to_categorical

import multiprocessing as mp

from nltk.tokenize import ToktokTokenizer

toktok = ToktokTokenizer()



#  other

import os

import json

from tqdm import tqdm

from copy import deepcopy



tqdm.pandas()



INPUT_DIR = '/kaggle/input/CORD-19-research-challenge/'
# from xhlulu's kernel



def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)



def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
data_path = INPUT_DIR + "biorxiv_medrxiv/biorxiv_medrxiv/"

files = load_files(data_path)

df = generate_clean_df(files)

df.head()
k=5



vocab = set(nltk.word_tokenize(" ".join(itertools.chain(df.text.tolist()))))

#word_index = {k: v for v, k in enumerate(vocab)}

print("Number of words:", len(vocab))

temp = df.text.progress_apply(lambda x: list(nltk.ngrams(nltk.sent_tokenize(" ".join(x.split("\n"))), k)))

all_grams = list(itertools.chain(*temp.tolist()))



def sentence_lengths(grams):

    lengths = []

    for gram in tqdm(grams):

        lengths.append(len(toktok.tokenize(" ".join(gram))))

    return pd.Series(lengths)



lengths = sentence_lengths(all_grams)

print(lengths.describe())

min_sent_length = lengths.min()
def encode_ngram_segment(gram, vocab_size, length):

    words = toktok.tokenize(" ".join(gram))

    start_idx = random.randrange(len(words) - length + 1)

    seq = words[start_idx:start_idx+length]

    tokens = hashing_trick(" ".join(seq), vocab_size, hash_function='md5', filters="") # shape (n,)

    return tokens



def pass_constants(*args):

    global n

    global query_length

    global all_grams

    n, query_length, all_grams = args



def encoder_worker(gram):

    global n

    global query_length

    global all_grams

    gram_encoding = int(hashlib.md5(gram.__repr__().encode()).hexdigest(), 16) % len(all_grams)

    if random.random() > 0.5:

        # then extract a random subset of words IN THE NGRAM

        blob_encoding = encode_ngram_segment(gram, n, query_length)

        return blob_encoding, gram_encoding, 1

    else:

        # then extract a random subset of words OUTSIDE the NGRAM

        neg_gram = random.choice(all_grams)

        blob_encoding = encode_ngram_segment(neg_gram, n, query_length)

        return blob_encoding, gram_encoding, 0



def generate_data(sentence_grams, query_length=min_sent_length, batch_size=16, n=len(vocab)):

    # all elements in sample with index i % neg_sample_size = 0 will be positive, neg. otherwise

    pool = mp.Pool(mp.cpu_count(), initializer=pass_constants, initargs=[n, query_length, all_grams])

    while True:

        sample = random.choices(sentence_grams, k=batch_size)

        num_grams = len(sentence_grams)

        blob_encodings, gram_encodings, targets = list(zip(*pool.map(encoder_worker, sample)))        

        yield ((np.array(blob_encodings), np.array(gram_encodings, ndmin=2)), np.array(targets))

    pool.close()
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Embedding, Dense, Input, Lambda, BatchNormalization, Activation, Dot

from tensorflow.keras.activations import tanh, sigmoid

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow.keras.backend as K



from sklearn.model_selection import train_test_split



tf.random.set_seed(42)
K.clear_session()



blob_input = Input((min_sent_length,))

gram_input = Input((1,))

blob_embeds = Embedding(len(vocab), 300)(blob_input)

mean_norm_sent_embeds = Lambda(lambda x: K.l2_normalize(K.mean(x, axis=1), axis=-1))(blob_embeds)

blob_proj = Dense(256)(mean_norm_sent_embeds)

bn = BatchNormalization(scale=False)(blob_proj)

activation = Activation(tanh)(bn)



gram_embeds = Embedding(len(all_grams), 256)(gram_input)

merged = Dot(-1)([activation, gram_embeds])

output = Activation(sigmoid)(merged)



model = Model(inputs=[blob_input, gram_input], outputs=output)



print(model.summary())



model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
X_train,X_val = train_test_split(all_grams, test_size=0.1, random_state=42)
checkpt = ModelCheckpoint("weights.{epoch:02d}.hdf5")

e_stopping = EarlyStopping(patience = 5)

batch_size = 16



train_gen = generate_data(X_train, batch_size=batch_size)

val_gen = generate_data(X_val, batch_size=batch_size)

history = model.fit_generator(train_gen, steps_per_epoch=int(np.ceil(len(X_train)/batch_size)), epochs=50, 

                              validation_data=val_gen, validation_steps=int(np.ceil(len(X_val)/batch_size)), callbacks=[checkpt, e_stopping],

                             use_multiprocessing=False)
!nvidia-smi