# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir())



import time

startTime = time.time()

# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import operator 

import re

import gc

import pickle



from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import StandardScaler
np.random.seed(201942)
def load_embed(file):

    def get_coefs(word, *arr):

        return word, np.asarray(arr, dtype='float32')



    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o) > 100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
def index_embs(embeddings_index, word_index, NUM_WORDS, fileName):

    

    all_embs = np.stack(embeddings_index.values())

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



#     nb_words = min(NUM_WORDS, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (NUM_WORDS, embed_size))



    for word, i in word_index.items():

        if i >= NUM_WORDS: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    

    np.save(fileName, embedding_matrix)

    print(fileName + " embedding matrix saved!")

    

    del(embeddings_index)

    del(word_index)

    del(embedding_matrix)

    gc.collect()
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



    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
def build_vocab(texts, num_words):

#     sentences = texts.apply(lambda x: x.split()).values

#     vocab = {}

#     for sentence in sentences:

#         for word in sentence:

#             try:

#                 vocab[word] += 1

#             except KeyError:

#                 vocab[word] = 1

    tokenizer = Tokenizer(num_words=NUM_WORDS, filters="")

    tokenizer.fit_on_texts(texts)

    

    # saving

    with open('tokenizer.pickle', 'wb') as handle:

        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

    print("\nKeras text tokenizer has been saved as tokenizer.pickle")

    

    return tokenizer.word_index
def add_lower(embedding, vocab):

    """

    Therer are words that are known with upper letters and unknown without. This method saves both variants of the word

    

    """



    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")
def known_contractions(embed): 

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
def get_mappings():

    """

    returns: 

    mispell_dict: mapping from mispelled word to correct word

    contraction_mapping: mapping from contraction to full word(s)

    punct_mapping: mapping from punctuation/special char to proper punctuation

    """

    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

    return mispell_dict, contraction_mapping, punct_mapping, punct
def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x
def add_features(df, textCol, outputFileName):

    # Create new DF for features

    df_features = pd.DataFrame()

    

    df_features['total_length'] = df[textCol].apply(len)

    df_features['capitals'] = df[textCol].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    df_features['caps_vs_length'] = df_features["capitals"] / df_features['total_length']

    df_features['num_words'] = df[textCol].str.count('\S+')

    df_features['num_unique_words'] = df[textCol].apply(lambda comment: len(set(w for w in comment.split())))

    df_features['words_vs_unique'] = df_features['num_unique_words'] / df_features['num_words']

    df_features = df_features.fillna(0)



    # Scale the features

    ss = StandardScaler()

    df_features = ss.fit_transform(df_features)

    

    print("\nFEATURES DF SAVED TO:")

    print("{0}_features.npy".format(outputFileName))

    np.save("{0}_features".format(outputFileName), df_features)
def run_text_preprocessing(df, textCol, outputFileName):

    mispell_map, contraction_map, punct_map, punct = get_mappings()

    

#     print("\nWorking on: " + outputFileName)

#     print("\n****** WORD COVERAGE BEFORE PREPROCESSING ******")

#     vocab = build_vocab(df[textCol])

#     print("Glove : ")

#     oov_glove = check_coverage(vocab, embed_glove)

#     print("Paragram : ")

#     oov_paragram = check_coverage(vocab, embed_paragram)

#     print("FastText : ")

#     oov_fasttext = check_coverage(vocab, embed_fasttext)

      

    # To lower char

    df['lowered_question'] = df[textCol].apply(lambda x: x.lower())

    # Contractions

    df['treated_question'] = df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_map))

    # Punct + Special chars

    df['treated_question'] = df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_map))

    # Mispelling

    #df['treated_question'] = df['treated_question'].apply(lambda x: correct_spelling(x, mispell_map))

    # Numbers

    df['treated_question'] = df['treated_question'].apply(lambda x: clean_numbers(x))

    

#     print("\n****** FINAL WORD COVERAGE AFTER PREPROCESSING ******")

#     vocab = build_vocab(df["treated_question"])

#     print("Glove : ")

#     oov_glove = check_coverage(vocab, embed_glove)

#     print("Paragram : ")

#     oov_paragram = check_coverage(vocab, embed_paragram)

#     print("FastText : ")

#     oov_fasttext = check_coverage(vocab, embed_fasttext)

    

    print("\nDATAFRAME SAVED TO:")

    print("{0}_processed_text.csv".format(outputFileName))

    df.to_csv("{0}_processed_text.csv".format(outputFileName))

    

    del(mispell_map)

    del(contraction_map) 

    del(punct_map)

    del(punct)

    gc.collect()

    

    return df
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
add_features(train, "question_text", "train")

add_features(test, "question_text", "test")
train = run_text_preprocessing(train, "question_text", "train")

test = run_text_preprocessing(test, "question_text", "test")
df_texts = pd.concat([train.drop('target', axis=1),test])

df_texts.to_csv("texts_processed.csv")
del(train)

del(test)

gc.collect()
glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'



NUM_WORDS = None

word_idx = build_vocab(df_texts["treated_question"], NUM_WORDS)

NUM_WORDS = len(word_idx) + 1

print(NUM_WORDS)



print("\n\n****** LOAD EMBEDDINGS ******")



print("\n---Extracting GloVe embedding---")

embed_glove = load_embed(glove)

index_embs(embed_glove, word_idx, NUM_WORDS, fileName="glove")

del(embed_glove)

gc.collect()



print("\n---Extracting Paragram embedding---")

embed_paragram = load_embed(paragram)

index_embs(embed_paragram, word_idx, NUM_WORDS, fileName="paragram")

del(embed_paragram)

gc.collect()



# print("\n---Extracting FastText embedding---")

# embed_fasttext = load_embed(wiki_news)

# index_embs(embed_fasttext, word_idx, NUM_WORDS, fileName="fasttext")

# del(embed_fasttext)

# gc.collect()
print(os.listdir())





(time.time() - startTime)/ 60
# dump all variables

import gc

gc.collect()

%reset -f
import pandas as pd

import numpy as np

import gc

import time

import pickle

import copy

import warnings

import math



from tensorflow import set_random_seed

import tensorflow as tf



from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from sklearn.metrics import classification_report, f1_score



from keras.engine.topology import Layer

from keras.utils.generic_utils import serialize_keras_object

from keras.utils.generic_utils import deserialize_keras_object

from keras.legacy import interfaces

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model

from keras import backend as K

from keras.layers import Input, Embedding, GRU, TimeDistributed, Dense, CuDNNGRU, Bidirectional, Dropout, SpatialDropout1D

from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization, CuDNNLSTM, Flatten

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.callbacks import Callback, ReduceLROnPlateau, LearningRateScheduler

from keras.optimizers import Optimizer

from keras import initializers

from keras import regularizers



import matplotlib.pyplot as plt



import time

startTimePt2 = time.time()
NUM_WORDS = 192111 #200549 

EMBEDDING_DIM = 300

NUM_FILTERS = 100

MAX_LEN = 72 #

BATCH_SIZE = 512

RANDOM_STATE = 201942

NUM_EPOCH = 6

LR = 0.003 # 3e-4

LR_MAX = LR * 6 # 7e-2

WD = 0.011 * (BATCH_SIZE / 979591 / NUM_EPOCH)**0.5 ## 979591  1044897

STEP_SIZE_CLR = 2 * (int(979591 * 2) / BATCH_SIZE)



# Suggested weight decay factor from the paper: w = w_norm * (b/B/T)**0.5

# b: batch size

# B: total number of training points per epoch

# T: total number of epochs

# w_norm: designed weight decay factor (w is the normalized one).
np.random.seed(RANDOM_STATE)

set_random_seed(RANDOM_STATE)
df_train = pd.read_csv("train_processed_text.csv")

df_test = pd.read_csv("test_processed_text.csv")

print("Train shape : ", df_train.shape)

print("Test shape : ", df_test.shape)
## fill up the missing values

X_train = df_train["treated_question"].fillna("_na_").values

X_test = df_test["treated_question"].fillna("_na_").values



y_train = df_train['target'].values
# Text Features

X_features_train = np.load("train_features.npy")

X_features_test = np.load("test_features.npy")
# loading

with open('tokenizer.pickle', 'rb') as handle:

    tokenizer = pickle.load(handle)
NUM_WORDS = len(tokenizer.word_index) + 1

NUM_WORDS
X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

print("hello word")
## Pad the sentences 

X_train = pad_sequences(X_train, maxlen=MAX_LEN)

X_test = pad_sequences(X_test, maxlen=MAX_LEN)

print("hello word")
X_train[0].shape
del(df_train)

del(df_test)
embedding_matrix_1 = np.load("glove.npy")

# embedding_matrix_2 = np.load("fasttext.npy")

embedding_matrix_3 = np.load("paragram.npy")

print("Done...")
embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis = 0)

np.shape(embedding_matrix)
del(embedding_matrix_1)

# del(embedding_matrix_2)

del(embedding_matrix_3)
gc.collect()

print("done")
class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



#         self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

#         self.history.setdefault('iterations', []).append(self.trn_iterations)



#         for k, v in logs.items():

#             self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())
def f1(y_true, y_pred):

    '''

    metric from here 

    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    '''

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        

        return recall

    

    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)



    return 2*((precision*recall)/(precision+recall+K.epsilon()))
class AdamW(Optimizer):

    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments

        lr: float >= 0. Learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        epsilon: float >= 0. Fuzz factor.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: float >= 0. Decoupled weight decay over each update.

    # References

        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

        - [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html)

        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/6)

                 epsilon=1e-8, decay=0., **kwargs):

        super(AdamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.init_lr = lr # decoupled weight decay (2/6)

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (3/6)

        self.epsilon = epsilon

        self.initial_decay = decay



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]

        wd = self.wd # decoupled weight decay (4/6)



        lr = self.lr

        if self.initial_decay > 0:

            lr *= (1. / (1. + self.decay * K.cast(self.iterations,

                                                  K.dtype(self.decay))))

        eta_t = lr / self.init_lr # decoupled weight decay (5/6)



        t = K.cast(self.iterations, K.floatx()) + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t)))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - eta_t * wd * p # decoupled weight decay (6/6)



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'weight_decay': float(K.get_value(self.wd)),

                  'epsilon': self.epsilon}

        base_config = super(AdamW, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
# Kernel

glorotInit = initializers.glorot_uniform(seed=RANDOM_STATE)

# Recurrent

orthoInit = initializers.Orthogonal(seed=RANDOM_STATE)
def squash(x, axis=-1):

    # s_squared_norm is really small

    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)

    # return scale * x

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



# A Capsule Implement with Pure Keras

class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)

            

    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer=glorotInit,

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer=glorotInit,

                                     trainable=True)



    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)

NUM_WORDS
def capsule_model():

    K.clear_session() 

    

    inp_features = Input(shape=(6,)) 

    

    inp = Input(shape=(MAX_LEN,))    

    emb = Embedding(NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(inp)

    emb_drop = SpatialDropout1D(rate=0.2)(emb)

    rnn = Bidirectional(CuDNNLSTM(75, return_sequences=True, 

                                kernel_initializer=glorotInit, recurrent_initializer=orthoInit))(emb_drop)



    caps = Capsule(num_capsule=4, dim_capsule=10, routings=8, share_weights=True)(rnn)

    flat = Flatten()(caps)



    dense_1 = Dense(100, activation="elu", kernel_initializer=glorotInit)(flat)

    

    concat = concatenate([dense_1, inp_features])

    

    out_drop = Dropout(0.12)(concat)

    out_bn = BatchNormalization()(out_drop)

    out = Dense(1, activation="sigmoid")(out_bn)

    

    model = Model(inputs=[inp_features, inp], outputs=out)

    model.compile(loss='binary_crossentropy', optimizer=AdamW(lr=LR, weight_decay=WD), metrics=["acc", f1])

    return model
def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    

    for threshold in [i * 0.01 for i in range(100)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result
def step_decay(epoch):

    initial_lrate = LR

    drop = 0.6

    epochs_drop = 5.0

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate
# clr = CyclicLR(base_lr=LR, max_lr=LR_MAX,

#                step_size=STEP_SIZE_CLR, mode='exp_range',

#                gamma=0.99994)

clr = CyclicLR(base_lr=0.001, max_lr=0.003,

               step_size=300., mode='exp_range',

               gamma=0.99994)
# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go

def train_pred(model, train_X,train_features_X, train_y, val_X, val_features_X, val_y, test_data, callbacks=None):

    st = time.time()      

    model.fit([train_features_X, np.array(train_X)], train_y, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, 

              validation_data=([val_features_X, np.array(val_X)], val_y), callbacks = callbacks, verbose=2)

    

    pred_val_y = model.predict([val_features_X, val_X], batch_size=1024, verbose=0)



    best_thresh = threshold_search(val_y, pred_val_y)

    

    print("\tVal F1 Score: {:.4f}\tThresh: {:.2f}".format(best_thresh["f1"], best_thresh["threshold"]))

    best_score = best_thresh["f1"]

    pred_test_y = model.predict(test_data, batch_size=1024, verbose=0)

    print("Training time was: " + str(time.time() - st))

    print('=' * 60)

    

    return pred_val_y, pred_test_y, best_score
train_meta = np.zeros(y_train.shape)

test_meta = np.zeros(len(X_test))



splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X_train, y_train))



for idx, (train_idx, valid_idx) in enumerate(splits):

    X_train_tmp = X_train[train_idx]

    X_features_train_tmp = X_features_train[train_idx]

    y_train_tmp = y_train[train_idx]

    

    X_val_tmp =  X_train[valid_idx]

    X_features_val_tmp = X_features_train[valid_idx]

    y_val_tmp = y_train[valid_idx]

    

    model = capsule_model()   

    #reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)

    #scheduleLR = LearningRateScheduler(schedule=step_decay, verbose=2)

    pred_val_y, pred_test_y, best_score = train_pred(model, X_train_tmp, X_features_train_tmp, y_train_tmp, X_val_tmp, 

                                                     X_features_val_tmp, y_val_tmp, [X_features_test ,X_test], 

                                                     callbacks=[clr])

    

    train_meta[valid_idx] = pred_val_y.reshape(-1)

    test_meta += pred_test_y.reshape(-1) / len(splits)



    del(X_train_tmp)

    del(y_train_tmp)

    del(X_val_tmp)

    del(y_val_tmp)

    gc.collect()
search_result = threshold_search(y_train, train_meta)

print(search_result)



sub = pd.read_csv('../input/sample_submission.csv')

sub.prediction = test_meta > search_result['threshold']

sub.to_csv("submission.csv", index=False)
(time.time() - startTimePt2) / 60