import time

import random

import pandas as pd

import numpy as np

import gc

import re

import torch

from torchtext import data

import spacy

from tqdm import tqdm_notebook, tnrange

from tqdm.auto import tqdm

import string

import math

import operator

import pkg_resources

from pyphen import Pyphen



tqdm.pandas(desc='Progress')

from collections import Counter

from textblob import TextBlob

from nltk import word_tokenize



import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable

from torchtext.data import Example

import torchtext

import os



from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten, GlobalAveragePooling1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



# cross validation and metrics

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import f1_score, accuracy_score, recall_score, mean_squared_error, r2_score

from torch.optim.optimizer import Optimizer

from unidecode import unidecode
# first try? the word2vec model
embed_size = 300 # how big is each word vector

max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 70 # max number of words in a question to use

batch_size = 512 # how many samples to process at once

n_epochs = 40 # how many times to iterate over all samples

n_splits = 5 # Number of K-fold Splits

SEED = 1000
import psutil

from multiprocessing import Pool

num_partitions = 8 # number of partitions to split dataframe

num_cores = psutil.cpu_count() # number of cores on your machine

print('number of cores: ', num_cores)
# function to run parallelly

def df_parallelize_run(df, func):

    df_split = np.array_split(df, num_partitions)

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df
def seed_everything(seed = 1029):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
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



def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known



def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        if s in text:

            text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



def correct_spelling(x, dic):

    for word in dic.keys():

        if word in text:

            x = x.replace(word, dic[word])

    return x



def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



def clean_special_chars(text, punct, mapping):

    for p in mapping:

        if p in text:

            text = text.replace(p, mapping[p])

    

    for p in punct:

        if p in text:

            text = text.replace(p, ' ' + str(p) + ' ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        if s in text:

            text = text.replace(s, specials[s])

    

    return text



def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print("Added " + str(count) + " words to embedding")   

    

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



def clean_text(x):

    x = str(x)

    for punct in puncts:

        if punct in x:

            x = x.replace(punct, ' ' + str(punct) + ' ')

    return x



mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)



#############################################

# add features in the datasets

#############################################

def add_features(df):

    df.loc[:, 'app_desc'] = df.loc[:,'app_desc'].progress_apply(lambda x:str(x))

    df.loc[:,'total_length'] = df.loc[:,'app_desc'].progress_apply(len)

#     df.loc[:,'capitals'] = df.loc[:,'app_desc'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))

#     df.loc[:,'caps_vs_length'] = df.loc[:,'app_desc'].progress_apply(lambda row: float(row['capitals'])/float(row['total_length']),

#                                 axis=1)

    df.loc[:,'num_words'] = df.app_desc.str.count('\S+')

    df.loc[:,'num_unique_words'] = df.loc[:,'app_desc'].progress_apply(lambda comment: len(set(w for w in comment.split())))

    df.loc[:,'words_vs_unique'] = df.loc[:,'num_unique_words'] / df['num_words']  

    df.loc[:,'size_bytes_in_MB'] = df.loc[:,'size_bytes'] / (1024 * 1024.0)

    df.loc[:,'isNotFree'] = df.loc[:,'price'].apply(lambda x: 1 if x > 0 else 0)

#     df.loc[:,'isNotGame'] = df.loc[:,'prime_genre'].apply(lambda x : 1 if x == 'Games' else 0)

    df.loc[:, 'rating_count_prev'] = df.loc[:, 'rating_count_tot'] - df.loc[:, 'rating_count_ver']

    df.loc[:, 'user_rating_prev'] = (df.loc[:, 'rating_count_tot'] * df.loc[:, 'user_rating'] - df.loc[:, 'rating_count_ver']*df.loc[:, 'user_rating_ver'])/df.loc[:, 'rating_count_prev']

#     feature_list = ['total_length', 'num_words', 'num_unique_words', 'rating_count_prev', 'caps_vs_length', 'size_bytes_in_MB', 'isNotGame', 'isNotFree']

#     feature_list = ['total_length', 'num_words', 'num_unique_words', 'rating_count_prev', 'user_rating_prev', 'user_rating_prev','caps_vs_length', 'size_bytes_in_MB', 'isNotFree']

    feature_list = ['total_length', 'num_words', 'num_unique_words']

    

    return df, feature_list
print(os.listdir('../input'))
#############################################

# load and preprocess the datasets

#############################################

target = 'user_rating_ver'

catagorical = 'prime_genre'



def prime_trans(x):

#     l = df.prime_genre.value_counts().index[:4]

    l = ['Games', 'Entertainment', 'Education', 'Photo & Video']

    if x in l:

        return x

    else:

        return "Other"

    

def load_and_prec():

    

    DATA_PATH = "../input"



    ## Data Read and Join

    data = pd.read_csv(os.path.join(DATA_PATH,"AppleStore.csv"))

    description_data = pd.read_csv(os.path.join(DATA_PATH,"appleStore_description.csv"))

    

    # Join the Data

    data = data.set_index("id")

    description_data = description_data.set_index("id")

    full_data = data.join(description_data,lsuffix='_left', rsuffix='')

    full_data.loc[:,"size_bytes"] = full_data["size_bytes_left"]

    del full_data["size_bytes_left"],full_data["track_name_left"],full_data["Unnamed: 0"]



    # split out the outliers

    full_data = full_data.loc[full_data[target] != 0]

    full_data = full_data.loc[full_data['price'] < 20.99]

    

     # select the features from the full dataset

    # total_features = [catagorical, 'size_bytes', 'price', 'rating_count_tot', 'rating_count_ver', 'sup_devices.num', 'ipadSc_urls.num', 'lang.num', target, 'app_desc']

    total_features = [catagorical, 'size_bytes', 'price', 'rating_count_tot', 'rating_count_ver', 'sup_devices.num', 'ipadSc_urls.num', 'lang.num','user_rating', target, 'app_desc']

    new_data = full_data.loc[:,total_features]

    #transform the prime genre and one-hot encoding it

    new_data.loc[:, catagorical] = new_data.loc[:, catagorical].progress_apply(lambda x: prime_trans(x))

    temp_data = new_data.loc[:, catagorical]

    temp_data = pd.get_dummies(new_data.loc[:, catagorical])

    new_data = new_data.join(temp_data)

    

    # lower

    new_data.loc[:,'app_desc'] = new_data['app_desc'].progress_apply(lambda x: x.lower())

    # clean the text

    new_data.loc[:,'app_desc'] = new_data['app_desc'].progress_apply(lambda x: clean_text(x))

    # clean numbers

    new_data.loc[:,'app_desc'] = new_data['app_desc'].progress_apply(lambda x: clean_numbers(x))

    # clean speelings

    new_data.loc[:,'app_desc'] = new_data['app_desc'].progress_apply(lambda x: replace_typical_misspell(x))

    # fill up the missing values

    new_data.loc[:,'app_desc'] = new_data['app_desc'].fillna("_##_").values

    

    ###################### Add Features ###############################

    new_data, feature_list = add_features(new_data)

    new_data = new_data.dropna()

    new_data = new_data.drop('user_rating', axis=1)

    y = new_data[[target]]

    cols = [i for i in new_data.columns if i not in [target]]

    X = new_data[cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    

    return full_data, new_data, X_train, X_test, y_train, y_test, feature_list



full_data, new_data, X_train, X_test, y_train, y_test, feature_list = load_and_prec()
import seaborn as sns

import matplotlib.pyplot as plt

cnt = full_data[target].value_counts().reindex(np.arange(0.0, 5.5, 0.5)).plot(kind='bar')
print(len(new_data.columns))

print(len(feature_list))

retA = [i for i in new_data.columns if i in feature_list]

retD = list(set(new_data.columns).difference(set(feature_list)))

print(retA)

print(retD)

print(new_data.columns)
f, ax = plt.subplots(figsize=(20, 20))

new_data_corr = new_data.corr()

sns.heatmap(new_data_corr,annot=True,linewidths=.5, fmt= '.2f',mask=np.zeros_like(new_data_corr, dtype=np.bool), cmap=sns.diverging_palette(400,500, as_cmap=True), square=True, ax=ax)

plt.show()
resample = 0 # 0 resampling, 1 no resampling



X_data = X_train.join(y_train)

count_max = np.max(y_train['user_rating_ver'].value_counts().values)



# for i in np.arange(1, 11, 1):

#     names =  locals()

#     names['index_class_'+str(i*5+5)] = X_data.loc[X_data['user_rating_ver'] == i*0.5+0.5]





###################################

# try the over sampling

###################################

def over_sampling():

    names = {}

    class_over = {}

    for i in np.arange(1, 10, 1):

        names['index_class_'+str(i*5+5)] = X_data.loc[X_data['user_rating_ver'] == i*0.5+0.5]    

        class_over['class_over_'+str(i*5+5)] = names['index_class_'+str(i*5+5)].sample(count_max, replace=True)

        if i == 1:

            train_over = class_over['class_over_'+str(i*5+5)]

        else:

            train_over = pd.concat([train_over, class_over['class_over_'+str(i*5+5)]], axis = 0)



    y_train = train_over[[target]]

    X_train = train_over.loc[:, train_over.columns != target]

    print(X_train.shape)

    print(y_train.shape)

    return X_train, y_train



###################################

# try the TomekLinks

###################################

# import imblearn

# from imblearn.under_sampling import TomekLinks

# def Tomek():

#     tl = TomekLinks(return_indices=True, ratio='majority')

#     X_tl, y_tl, id_tl = tl.fit_sample(X_train, y_train)

#     print(X_t1.shape())

#     print(y_t1.shape())

if resample == 0:  

    X_train, y_train = over_sampling()
check = ['size_bytes', 'rating_count_tot', 'rating_count_ver', catagorical, target, 'app_desc']

feature_cols_combinated = [i for i in new_data.columns if i not in check]

X_train = X_train.loc[:,feature_cols_combinated].fillna(0)

X_test = X_test.loc[:,feature_cols_combinated].fillna(0)
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



rf=RandomForestRegressor(random_state=42)

rf.fit(X_train,y_train)

# predict the metrology

y_pred_rf= rf.predict(X_test)

# print the r^2 score

print("r2_score:",metrics.r2_score(y_test, y_pred_rf))

print("MSE:",metrics.mean_squared_error(y_test,y_pred_rf))