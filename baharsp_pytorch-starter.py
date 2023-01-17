import re

import time

import gc

import random

import os



import numpy as np

import pandas as pd



from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



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

from keras.engine import InputSpec, Layer





import torch

import torch.nn as nn

import torch.utils.data



tqdm.pandas()

from nltk.corpus import stopwords

stop = stopwords.words('english')
embed_size = 300 # how big is each word vector

max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)

max_length = 60 # max number of words in a question to use



SEED = 1029





embedding_size = 600

learning_rate = 0.001

batch_size = 512

num_epoch = 4
def seed_torch(seed=1029):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
def seed_torch(seed=1029):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):

    mispell_re = re.compile(('(%s)' % '|'.join(mispell_dict.keys())))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)





def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
def load_and_prec():

    train_df = pd.read_csv("/kaggle/input/train.csv")

    test_df = pd.read_csv("/kaggle/input/test.csv")

    print("Train shape : ",train_df.shape)

    print("Test shape : ",test_df.shape)

    

    # lower

    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())

    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())

    

    # Clean the text

    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))

    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))

    

    # Clean numbers

    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))

    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))



    # Clean speelings

    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))



#     # Remove stopwords

#     train_df["question_text"] = train_df["question_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#     test_df["question_text"] = test_df["question_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    

    ## fill up the missing values

    train_X = train_df["question_text"].fillna("_##_").values

    test_X = test_df["question_text"].fillna("_##_").values



    ## Tokenize the sentences

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(train_X))

    train_X = tokenizer.texts_to_sequences(train_X)

    test_X = tokenizer.texts_to_sequences(test_X)



    ## Pad the sentences 

    train_X = pad_sequences(train_X, maxlen=max_length)

    test_X = pad_sequences(test_X, maxlen=max_length)



    ## Get the target values

    train_y = train_df['target'].values

    

    #shuffling the data

    np.random.seed(SEED)

    trn_idx = np.random.permutation(len(train_X))



    train_X = train_X[trn_idx]

    train_y = train_y[trn_idx]

    

    return train_X, test_X, train_y, tokenizer.word_index
def load_glove(word_index):

    EMBEDDING_FILE = '/kaggle/input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 



def load_para(word_index):

    embedding_size = 300

    EMBEDDING_FILE = '/kaggle/input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    

    return embedding_matrix 



def load_wiki(word_index):

    EMBEDDING_FILE = '/kaggle/input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix
def build_model(embedding_matrix, max_features, embedding_size=300):

    inp = Input(shape=(max_length,))

    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(inp)

    x = SpatialDropout1D(0.3)(x)

    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)

    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)

    max_pool1 = GlobalMaxPooling1D()(x1)

    max_pool2 = GlobalMaxPooling1D()(x2)

    conc = Concatenate()([max_pool1, max_pool2])

    predictions = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=predictions)

    adam = optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model
start_time = time.time()

train_X, test_X, train_y, word_index = load_and_prec()



pred_prob = np.zeros((len(test_X),), dtype=np.float32)



print("Loading embedding matrix glove and wiki...")

embedding_matrix_wiki = load_wiki(word_index)

embedding_matrix_glove = load_glove(word_index)

total_time = (time.time() - start_time) / 60

print("Took {:.2f} minutes".format(total_time))

embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_wiki), axis=1)

start_time = time.time()

print("Start training first...")

model = build_model(embedding_matrix, max_features, embedding_size)

model.fit(train_X, train_y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)

pred_prob += 0.15*np.squeeze(model.predict(test_X, batch_size=batch_size, verbose=2))

model.fit(train_X, train_y, batch_size=batch_size, epochs=1, verbose=2)

pred_prob += 0.35*np.squeeze(model.predict(test_X, batch_size=batch_size, verbose=2))

del model, embedding_matrix_wiki, embedding_matrix

gc.collect()

K.clear_session()

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Loading embedding matrix para...")

embedding_matrix_para = load_para(word_index)



embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_para), axis=1)

start_time = time.time()

print("Start training second...")

model = build_model(embedding_matrix, max_features, embedding_size)

print('end build model')

model.fit(train_X, train_y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)

pred_prob += 0.15*np.squeeze(model.predict(test_X, batch_size=batch_size, verbose=2))

model.fit(train_X,train_y, batch_size=batch_size, epochs=1, verbose=2)

pred_prob += 0.35*np.squeeze(model.predict(test_X, batch_size=batch_size, verbose=2))

print("Took {:.2f} minutes".format((time.time() - start_time) / 60))
test = pd.read_csv('../input/test.csv').fillna(' ')

submission = pd.DataFrame.from_dict({'qid': test['qid']})

submission['prediction'] = (pred_prob>0.35).astype(int)

submission.to_csv('submission.csv', index=False)

del model, embedding_matrix_para, embedding_matrix

gc.collect()

K.clear_session()

print("--- %s seconds ---" % (time.time() - start_time))
!head submission.csv