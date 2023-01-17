# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from keras.engine.topology import Layer

import math

import operator 

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, TimeDistributed, CuDNNLSTM,Conv2D, SpatialDropout1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, Flatten, Reshape, AveragePooling2D, Average, BatchNormalization

from keras.models import Model

from keras.layers import Wrapper

import keras.backend as K

from keras.optimizers import Adam

from keras import initializers, regularizers, constraints, optimizers, layers

import re

import gc

from sklearn.preprocessing import StandardScaler

tqdm.pandas()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",

                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",

                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                       "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have",

                       "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",

                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",

                       "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",

                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",

                       "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",

                       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",

                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}



contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),

                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'),

                        (r'dont', 'do not'), (r'wont', 'will not') ]



punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



mispell_dict = {'advanatges': 'advantages', 'irrationaol': 'irrational' , 'defferences': 'differences','lamboghini':'lamborghini','hypothical':'hypothetical', 'colour': 'color',

                'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'qoura' : 'quora', 'sallary': 'salary', 'Whta': 'What',

                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',

                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',

                'pennis': 'penis', 'Etherium': 'Ethereum', 'etherium': 'ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota',

                'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',

                'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}

def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text

def replaceContraction(text):

    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]

    for (pattern, repl) in patterns:

        (text, count) = re.subn(pattern, repl, text)

    return text

def clean_text(x):

    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', ' number ', x)

    x = re.sub('[0-9]{4}', ' number ', x)

    x = re.sub('[0-9]{3}', ' number ', x)

    x = re.sub('[0-9]{2}', ' number ', x)

    return x

def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    for p in punct:

        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters

    for s in specials:

        text = text.replace(s, specials[s])

    return text

def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
# Un vistazo a los datos

train_df[train_df.target==1].head()
def statistical_features(df):

    stat_features = pd.DataFrame()

    stat_features['txt_len'] = df['question_text'].progress_apply(lambda x: len(str(x))) # incluye espacios

    stat_features['word_count'] = df['question_text'].progress_apply(lambda x: len(str(x).split(" ")))

    stat_features['!_count'] = df['question_text'].progress_apply(lambda x: x.count('!'))

    stat_features['?_count'] = df['question_text'].progress_apply(lambda x: x.count('?'))

    stat_features['upper_word_count'] = df['question_text'].progress_apply(lambda x: len([x for x in x.split() if x.isupper()]))

    stat_features['unique_word_count'] = df['question_text'].progress_apply(lambda x: len(set(x.split())))

    return stat_features
#train_df = statistical_features(train_df)

#train_df[train_df.target==1].head()
#test_df = statistical_features(test_df)

#test_df.head()
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

print("Extrayendo GloVe embedding...")

embed_glove = load_embed(glove)
#print("Extrayendo Paragram embedding...")

#embed_paragram = load_embed(paragram)
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab

def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Se agregaron {count} palabras al embedding")
df = pd.concat([train_df ,test_df])

vocab = build_vocab(df['question_text'])
print("GloVe v2: ")

add_lower(embed_glove, vocab)

#oov_glove = check_coverage(vocab_low, embed_glove)
#print("Paragram v2: ")

#add_lower(embed_paragram, vocab)

#oov_paragram = check_coverage(vocab_low, embed_paragram)
# Llevar a minúsculas

train_df['treated_question'] = train_df['question_text'].progress_apply(lambda x: x.lower())

# Contracciones

train_df['treated_question'] = train_df['treated_question'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

# Caracteres especiales

train_df['treated_question'] = train_df['treated_question'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# Corregir ortografía

train_df['treated_question'] = train_df['treated_question'].progress_apply(lambda x: correct_spelling(x, mispell_dict))



# Llevar a minúsculas

test_df['treated_question'] = test_df['question_text'].progress_apply(lambda x: x.lower())

# Contracciones

test_df['treated_question'] = test_df['treated_question'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

# Caracteres especiales

test_df['treated_question'] = test_df['treated_question'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# Corregir ortografía

test_df['treated_question'] = test_df['treated_question'].progress_apply(lambda x: correct_spelling(x, mispell_dict))
## some config values 

embed_size = 300 # how big is each word vector

max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)

max_len = 100 # max number of words in a question to use
def make_data(X, X_test):

    t = Tokenizer(num_words=max_features, filters='', oov_token='<OOV>')

    t.fit_on_texts(X)

    X = t.texts_to_sequences(X)

    X_test = t.texts_to_sequences(X_test)

    X = pad_sequences(X, maxlen=max_len)

    X_test = pad_sequences(X_test, maxlen=max_len)

    return X, X_test, t.word_index
X, X_test, word_index = make_data(train_df['treated_question'], test_df['treated_question'])
y = train_df['target'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
def make_embed_matrix(embeddings_index, word_index, len_voc):

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]

    word_index = word_index

    nb_words = min(len_voc, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= len_voc:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: 

            embedding_matrix[i] = embedding_vector

    return embedding_matrix
embedding_matrix_glove = make_embed_matrix(embed_glove, word_index, max_features)

del word_index

del embed_glove

gc.collect()
print("Extrayendo Paragram embedding...")

embed_paragram = load_embed(paragram)
#print("Paragram v2: ")

add_lower(embed_paragram, vocab)

#oov_paragram = check_coverage(vocab_low, embed_paragram)
_, _, word_index = make_data(train_df['treated_question'], test_df['treated_question'])

embedding_matrix_paragram = make_embed_matrix(embed_paragram, word_index, max_features)

del word_index

del embed_paragram

gc.collect()
embedding_matrix_final = np.sum([embedding_matrix_glove*0.7,embedding_matrix_paragram*0.3], axis=0)
train_stats = statistical_features(train_df)

train_stats[train_df.target==1].head()
test_stats = statistical_features(test_df)

test_stats.head()
# Normalización de statistic features

sc = StandardScaler()

train_stats = sc.fit_transform(train_stats)

test_stats = sc.transform(test_stats)
y = train_df['target'].values

X_train_stats, X_val_stats, _, _ = train_test_split(train_stats, y, test_size=0.1, random_state=42)
def model():

    optim = Adam(lr=0.0010)

    

    main_inp = Input(shape=(max_len,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix_final], trainable=True)(main_inp)

    x = SpatialDropout1D(0.15)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

    x = Conv1D(filters=64, kernel_size=1)(x)

    x = GlobalMaxPool1D()(x)

    

    stat_inp = Input(shape=(X_train_stats.shape[1],))

    x_2 = Dense(64, activation="relu")(stat_inp)

    x_2 = Dropout(0.15)(x_2)

    x_2 = Dense(32, activation="relu")(x_2)

    

    x_f = concatenate([x, x_2])

    x_f = Dense(128, activation="relu")(x_f)

    x_f = Dropout(0.15)(x_f)

    x_f = BatchNormalization()(x_f)

    x_f = Dense(1, activation="sigmoid")(x_f)

    

    model = Model(inputs=[main_inp, stat_inp], outputs = x_f)

    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['binary_accuracy'])

    return model
model1 = model()

print(model1.summary())
BATCH_SIZE = 512

EPOCHS = 2
model1.fit([X_train,X_train_stats], y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([X_val,X_val_stats],y_val))
def f1_score(pred_model):

    max_t = 0

    max_f1 = 0

    for thresh in np.arange(0.1, 0.701, 0.01):

        thresh = np.round(thresh, 2)

        f1 = metrics.f1_score(y_val, (pred_model>thresh).astype(int))

        #print("F1 score at threshold {0} is {1}".format(thresh, f1))

        if(f1>max_f1):

            max_f1 = f1

            max_t = thresh

    print(max_t, max_f1) 

    return max_t
pred_model1 = model1.predict([X_val,X_val_stats], batch_size=1024, verbose=1)

f1_score(pred_model1)
model2 = model()

model2.fit([X_train,X_train_stats], y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([X_val,X_val_stats],y_val))
pred_model2 = model2.predict([X_val,X_val_stats], batch_size=1024, verbose=1)

f1_score(pred_model2)
pred_model = pred_model1*0.5 + pred_model2*0.5

max_t = f1_score(pred_model)
pred_model1_test = model1.predict([X_test,test_stats], batch_size=1024, verbose=1)

pred_model2_test = model2.predict([X_test,test_stats], batch_size=1024, verbose=1)

pred_model_test = pred_model1_test*0.5 + pred_model2_test*0.5

pred_test = (pred_model_test>max_t).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test

out_df.to_csv("submission.csv", index=False)