# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import numpy as np 
import pandas as pd 
from numpy import array,asarray
import re
from tqdm import tqdm
from keras.regularizers import l1,l2
import os, gc
import random
from nltk.corpus import brown
import nltk
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from keras_tqdm import TQDMNotebookCallback
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer, MWETokenizer
import time
import pyLDAvis
from pyLDAvis.sklearn import prepare
from sklearn.model_selection import train_test_split
import keras
import keras.backend as K
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Conv1D, Dropout, Embedding, SpatialDropout1D, Bidirectional, CuDNNGRU,CuDNNLSTM, GRU, Dense, Concatenate, Reshape, Dot, Softmax, Activation, RepeatVector, Lambda, Conv1D, GlobalMaxPool1D, GlobalAvgPool1D, MaxPool1D, BatchNormalization, LSTM
from keras.layers.merge import concatenate
from gensim.models import KeyedVectors
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import f1_score
from keras.layers import GaussianDropout
df = pd.read_csv('../input/train.csv')
df.columns

parser = English()
def cleaner(doc):
    return ' '.join([token.lemma_ for token in parser(doc.lower()) if token.lemma_ not in STOP_WORDS and token.lemma_ != "-PRON-" and token.lemma_ not in punctuation])

tqdm.pandas()
df['clean'] = df['question_text'].progress_apply(cleaner)
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def cleaner_2(doc):
    doc = str(doc)
    # cleaning numbers
    doc = re.sub('[0-9]{5,}', '#####', doc)
    doc = re.sub('[0-9]{4}', '####', doc)
    doc = re.sub('[0-9]{3}', '###', doc)
    doc = re.sub('[0-9]{2}', '##', doc)
    
    # spacing out punctuations
    for punct in puncts:
        doc = doc.replace(punct, f' {punct} ')
    return doc

tqdm.pandas()
df['clean_2'] = df['question_text'].progress_apply(cleaner_2)
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'cant': 'cannot',
                'couldnt': 'could not',
                'shant':'shall not',
                'shouldnt': 'should not',
                'wont': 'will not',
                'wouldnt': 'would not',
                'havent': 'have not',
                'hadnt':'had not',
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
tqdm.pandas()
df["clean_2"] = df["clean_2"].progress_apply(lambda x: replace_typical_misspell(x))
tick = time.time()
brown_vocab_lower = brown.words(categories=brown.categories())
brown_vocab_lower = set(map(lambda x:x.lower(),brown_vocab_lower))
print("brown vocab generated in {} seconds.".format(str(time.time()-tick)))

tick = time.time()
EMBEDDING_FILE = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
def get_coefs(word,*arr): return word
glove_vocab = set(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))
glove_vocab = set(map(lambda x:x.lower(),glove_vocab))
print("glove vocab generated in {} seconds.".format(str(time.time()-tick)))


tick = time.time()
EMBEDDING_FILE = "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"
def get_coefs(word,*arr): return word
wiki_vocab = set(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))
wiki_vocab = set(map(lambda x:x.lower(),wiki_vocab))
print("wiki vocab generated in {} seconds.".format(str(time.time()-tick)))


tick = time.time()
EMBEDDING_FILE = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
def get_coefs(word,*arr): return word
paragram_vocab = set(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8', errors='ignore')if len(o)>100)
paragram_vocab = set(map(lambda x:x.lower(),paragram_vocab))
print("paragram vocab generated in {} seconds.".format(str(time.time()-tick)))
final_vocab = set(list(glove_vocab) + list(wiki_vocab) + list(brown_vocab_lower) + list(paragram_vocab))
del glove_vocab,wiki_vocab,brown_vocab_lower,paragram_vocab, EMBEDDING_FILE
gc.collect()
def get_feats(doc):
    
    token_length = len(doc.split())
    i1 = len(re.findall(r'(?=\bI\b|\bwe\b|\bme\b|\bus\b|\bmy\b|\bour\b|\bmine\b|\bours\b)', doc))/token_length # FP
    i2 = len(re.findall(r'(?=\byou\b|\byour\b|\byours\b)', doc)) /token_length # sP
    i3 = len(re.findall(r'(?=\bhe\b|\bshe\b|\bit\b|\bhim\b|\bher\b|\bhis\b|\bhers\b|\bits\b)', doc))/token_length # tps
    i4 = len(re.findall(r'(?=\bthey\b|\bthem\b|\btheir\b|\btheirs\b)', doc))/token_length # tpp
    i5 = len(re.findall(r'(?=\bwhat\b|\bwhatever\b|\bwhich\b|\bwhichever\b|\bwho\b|\bwhoever\b|\bwhom\b|\bwhomever\b|\bwhoses\b)', doc)) # interrogative
    i6 = len(re.findall(r'(?=\bmyself\b|\byourself\b|\bherself\b|\bhimself\b|\bitself\b|\bourselves\b|\byourselves\b|\bthemselves\b)', doc)) # intensive
    i7 = len(re.findall(r'(?=\bfor\b|\band\b|\bnor\b|\bbut\b|\byet\b|\bso\b|\bbefore\b|\bonce\b|\bsince\b|\bthough\b|\bwhile\b|\bas\b|\bbecause\b|\bafter\b)', doc))/token_length # conjunctions
    
    i8 = (len(doc) - len( re.findall('[a-zA-Z]', doc)) - doc.count(' ') - len(re.findall('[0-9]', doc)) - doc.count(',') - doc.count('?') - doc.count('.'))
    i9 = doc.count(',')
    i10 = doc.count('?')
    i11 = token_length
    # negs
    i12 = len(re.findall(r'(?=\bcan\'t\b|\bcouldn\'t\b|\bshan\'t\b|\bshouldn\'t\b|\bwouldn\'t\b|\bhaven\'t\b|\bdidn\'t\b|\bnot\b|\bnever\b|\bwon\'t\b|\bdon\'t\b|\bhadn\'t\b|\bcant\b|\bcouldnt\b|\bshant\b|\bshouldnt\b|\bwouldnt\b|\bhavent\b|\bdidnt\b|\bwont\b|\bdont\b|\bhadnt\b)', doc))
    
    #spelling mistakes
    temp_doc = (" ").join(re.findall(r"[a-zA-Z0-9\']+", doc))
    i13 = len(set(temp_doc.split())-final_vocab)
    
    return (i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13)
tqdm.pandas()
df['feat_vect'] = df['question_text'].progress_apply(get_feats)
# tqdm.pandas()
# df["num_words_upper"] = df["question_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# ## Number of title case words in the text
# tqdm.pandas()
# df["num_words_title"] = df["question_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# ## Average length of the words in the text
# tqdm.pandas()
# df["mean_word_len"] = df["question_text"].progress_apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df['fp'] = df['feat_vect'].apply(lambda x: x[0])
df['sp'] = df['feat_vect'].apply(lambda x: x[1])
df['tps'] = df['feat_vect'].apply(lambda x: x[2])
df['tpp'] = df['feat_vect'].apply(lambda x: x[3])
# df['interrogative'] = df['feat_vect'].apply(lambda x: x[4])
# df['intensive'] = df['feat_vect'].apply(lambda x: x[5])
df['conjunction'] = df['feat_vect'].apply(lambda x: x[6])
df['special_chars'] = df['feat_vect'].apply(lambda x: x[7])
df['commas'] = df['feat_vect'].apply(lambda x: x[8])
df['qm'] = df['feat_vect'].apply(lambda x: x[9])
df['len'] = df['feat_vect'].apply(lambda x: x[10])
# df['negs'] = df['feat_vect'].apply(lambda x: x[11])
df['sm'] = df['feat_vect'].apply(lambda x: x[12])
sincere = df[df['target']==0]
insincere = df[df['target']==1]
sincere.head()
insincere.head()
# Prepare test data
test = pd.read_csv('../input/test.csv')
test['clean'] = test['question_text'].apply(cleaner)
test['clean_2'] = test['question_text'].apply(cleaner_2)
# tqdm.pandas()
# test["num_words_upper"] = test["question_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# ## Number of title case words in the text
# tqdm.pandas()
# test["num_words_title"] = test["question_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# ## Average length of the words in the text
# tqdm.pandas()
# test["'mean_word_len'"] = test["question_text"].progress_apply(lambda x: np.mean([len(w) for w in str(x).split()]))
tqdm.pandas()
test['feat_vect'] = test['question_text'].progress_apply(get_feats)

test['fp'] = test['feat_vect'].apply(lambda x: x[0])
test['sp'] = test['feat_vect'].apply(lambda x: x[1])
test['tps'] = test['feat_vect'].apply(lambda x: x[2])
test['tpp'] = test['feat_vect'].apply(lambda x: x[3])
# test['interrogative'] = test['feat_vect'].apply(lambda x: x[4])
# test['intensive'] = test['feat_vect'].apply(lambda x: x[5])
test['conjunction'] = test['feat_vect'].apply(lambda x: x[6])
test['special_chars'] = test['feat_vect'].apply(lambda x: x[7])
test['commas'] = test['feat_vect'].apply(lambda x: x[8])
test['qm'] = test['feat_vect'].apply(lambda x: x[9])
test['len'] = test['feat_vect'].apply(lambda x: x[10])
# test['negs'] = test['feat_vect'].apply(lambda x: x[11])
test['sm'] = test['feat_vect'].apply(lambda x: x[12])
test.head()
del final_vocab
gc.collect()
test.head()
### LDA v2 - only bigrams & trigrams

def my_tweet_tokenizer(doc):
    t1 = TweetTokenizer()
    return t1.tokenize(doc)
c2 = CountVectorizer( tokenizer=my_tweet_tokenizer, ngram_range=(2,3), min_df=6)

tick = time.time()
c2.fit(insincere['clean'])
bow2 = c2.transform(insincere['clean'])
print('Time taken to fit CV: ' + str(time.time()-tick) + ' seconds.')

lda2 = LatentDirichletAllocation(n_components=8, learning_method='batch')
tick = time.time()
lda2.fit(bow2)
print('Time taken to fit lda: ' + str(time.time()-tick) + ' seconds.')

# pyLDAvis.enable_notebook()
# prepare(lda2,bow2,c2)
# Prepare training data: 0.8 of sincere, 1 of sincere
tick = time.time()
new_train_df = pd.concat([sincere.sample(frac=0.8, random_state=42),insincere])
new_train_df = new_train_df.sample(frac=1, random_state=113)

# Train test split
X_train, X_dev, y_train, y_dev = train_test_split(new_train_df, new_train_df['target'], test_size=0.2, random_state=42, stratify=new_train_df['target'])
t1 = Tokenizer()
t1.fit_on_texts(pd.concat([df['clean_2'], test['clean_2']]))

t2 = Tokenizer(filters=None, char_level=True)
t2.fit_on_texts(pd.concat([df['clean_2'], test['clean_2']]))

print('Time taken to fit keras tokenizers : ' + str(time.time()-tick) + ' seconds.')
tick = time.time()
X_train_tts = t1.texts_to_sequences(X_train['clean_2'])
X_dev_tts = t1.texts_to_sequences(X_dev['clean_2'])
test_tts = t1.texts_to_sequences(test['clean_2'])

X_train_tts = pad_sequences(X_train_tts, maxlen=90, padding='post', truncating='post')
X_dev_tts = pad_sequences(X_dev_tts, maxlen=90, padding='post', truncating='post')
test_tts = pad_sequences(test_tts, maxlen=90, padding='post', truncating='post')

X_train_cts = t2.texts_to_sequences(X_train['clean_2'])
X_dev_cts = t2.texts_to_sequences(X_dev['clean_2'])
test_cts = t2.texts_to_sequences(test['clean_2'])

X_train_cts = pad_sequences(X_train_cts, maxlen=180, padding='post', truncating='post')
X_dev_cts = pad_sequences(X_dev_cts, maxlen=180, padding='post', truncating='post')
test_cts = pad_sequences(test_cts, maxlen=180, padding='post', truncating='post')

print('Time taken for tts conversion : ' + str(time.time()-tick) + ' seconds.')
tick = time.time()
bow_train = c2.transform(X_train['clean'])
ldavec_train = lda2.transform(bow_train)

bow_dev = c2.transform(X_dev['clean'])
ldavec_dev = lda2.transform(bow_dev)

bow_test = c2.transform(test['clean'])
ldavec_test = lda2.transform(bow_test)
print('Time taken to generate lda vectors for train, dev & test : ' + str(time.time()-tick) + ' seconds.')
feat_train = X_train[['fp', 'sp', 'tps', 'tpp',  'conjunction', 'special_chars', 
                      'commas', 'qm', 'len',  'sm']].values
feat_dev = X_dev[['fp', 'sp', 'tps', 'tpp', 'conjunction', 'special_chars', 
                      'commas', 'qm', 'len', 'sm']].values
feat_test = test[['fp', 'sp', 'tps', 'tpp', 'conjunction', 'special_chars', 
                      'commas', 'qm', 'len',  'sm']].values
tick = time.time()
embedding_dim = 300
vocab_size = len(t1.word_index)+1
# vocab_size = 100000
EMBEDDING_FILE = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_glove = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))

embedding_matrix = np.random.randn(vocab_size,embedding_dim).astype(np.float32) * np.sqrt(2.0/vocab_size)

for word, i in t1.word_index.items():
    embedding_vector = embeddings_glove.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector

del embeddings_glove, EMBEDDING_FILE
gc.collect()
print("Embedding Matrix generated in {} seconds.".format(str(time.time()-tick)))
tick = time.time()
EMBEDDING_FILE = "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_wiki = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))

embedding_matrix_wiki = np.random.randn(vocab_size,embedding_dim).astype(np.float32) * np.sqrt(2.0/vocab_size)

for word, i in t1.word_index.items():
    embedding_vector = embeddings_wiki.get(word)
    if embedding_vector is not None: 
        embedding_matrix_wiki[i] = embedding_vector

del embeddings_wiki, EMBEDDING_FILE
gc.collect()
print("Embedding Matrix WIKI generated in {} seconds.".format(str(time.time()-tick)))
tick = time.time()
from gensim.models import KeyedVectors
embeddings_google = KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
print("Load time {} seconds.".format(str(time.time()-tick)))
embedding_matrix_google = np.random.randn(vocab_size,embedding_dim).astype(np.float32) * np.sqrt(2.0/vocab_size)
tick = time.time()
for word, i in t1.word_index.items():
    try:
        embedding_vector = 'init'
        embedding_vector = embeddings_google[word]
        embedding_matrix_google[i] = embedding_vector
    except:
        if embedding_vector=='init':
             pass
print("Embedding Matrix google generated in {} seconds.".format(str(time.time()-tick)))
del embeddings_google
gc.collect()
mean_emb = np.mean([embedding_matrix, embedding_matrix_wiki, embedding_matrix_google], axis=0)
mean_emb.shape
char_embedding_matrix = np.random.randn(len(t2.word_index)+1,128).astype(np.float32) * np.sqrt(2.0/len(t2.word_index)+1)
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

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
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


def f1(y_true, y_pred):
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
class Configuration:
    
    def get_config_object():
        config = Configuration()
        config.max_seq_size = 90
        config.word_vocab_size = vocab_size
        config.embedding_dim = 300
        config.units1 =64 
        config.units2 = 32
        config.units3 = 32
        
        return config
class Model:
    
    def __init__(self, config):
        self.config = config
        
    def get_model(self):
        
        input_layer = Input(batch_shape=(None,self.config.max_seq_size), name='input_layer')
        input_layer_2 = Input(batch_shape=(None,180), name='input_layer_cts')
        input_layer_3 = Input(batch_shape=(None,10), name='input_layer_feat')
        lda_input = Input(batch_shape=(None, 8), name='lda_input_layer')
        
        embed_op = Embedding(input_dim= self.config.word_vocab_size, input_length=self.config.max_seq_size, output_dim=self.config.embedding_dim, 
                              weights=[mean_emb], trainable=True, name='word_embeddings')(input_layer)
#         
        spdropout = SpatialDropout1D(0.2, name='sp_dropout')(embed_op)
        bigru1 = Bidirectional(CuDNNLSTM(units=self.config.units1, return_sequences=True, name='bigru1'))(spdropout)
#         spdropout_lstm = SpatialDropout1D(0.1, name='sp_dropout_lstm')(bigru1)
        
        embed_op_2 = Embedding(input_dim=len(t2.word_index)+1, input_length=180, output_dim=128, name='char_embeddings')(input_layer_2)
        spdropout_2 = SpatialDropout1D(0.2, name='sp_dropout_2')(embed_op_2)
        conv1 = Conv1D(filters=32, kernel_size=2, name='conv1')(spdropout_2)
        maxpool1 = GlobalMaxPool1D()(conv1)
#         avgpool1 = GlobalAvgPool1D()(conv1)
        
        conv2 = Conv1D(filters=32, kernel_size=4, name='conv2')(spdropout_2)
        maxpool2 = GlobalMaxPool1D()(conv2)
#         avgpool2 = GlobalAvgPool1D()(conv2)
        
        conv3 = Conv1D(filters=32, kernel_size=8, name='conv3')(spdropout_2)
        maxpool3 = GlobalMaxPool1D()(conv3)
#         avgpool3 = GlobalAvgPool1D()(conv3)

        # Attention
        # Inputs: [bs, TX, units2], Output = [bs,TX, dense_units]
        tile_layer = RepeatVector(n=90, name='tile_layer')
        tiled_op = tile_layer(lda_input)
        concat_1 = Concatenate(axis=2, name='concat_1')([bigru1,tiled_op])
        
        att_dense_layer = Dense(units=1, name='attention_dense_1', activation='tanh')
        att_dense_op_1 = att_dense_layer(concat_1) #Output = [bs,TX, 1]
        softmax_layer = Softmax(axis=1, name='softmax_over_time')
        alpha_1 = softmax_layer(att_dense_op_1) #Output = [bs,TX, 1]
        dot_layer = Dot(axes=1, name='dot_layer')
        context_1 = dot_layer([alpha_1, concat_1])
        context_reshape_1 = Reshape(target_shape=(136,), name='reshape_context')(context_1)#133 136 150
       
        
        selector = Lambda(lambda x: x[:, -1], name='slicer')
        lt_bigru1 = selector(bigru1)
        
      
        concat = concatenate([lt_bigru1,context_reshape_1,maxpool1,maxpool2, maxpool3,lda_input, input_layer_3], name='concatenate')
        
        dense1 = Dense(units=self.config.units3, activation='tanh', name='dense1')(concat)
#         drop1 = Dropout(0.1, name='Dropout_1_c3')(dense1)
        drop1 = GaussianDropout(0.1, name='Dropout_1_c3')(dense1)
        bn = BatchNormalization()(drop1)
        dense2 = Dense(units=16, activation='tanh', name='dense2')(bn)
        dense3 = Dense(units=1, activation='sigmoid', name='output')(dense2)
        classifier = keras.models.Model(inputs=[input_layer,input_layer_2,input_layer_3,lda_input], outputs=dense3)
        
        return classifier
    
        
    def get_model_2(self):
        
        input_layer = Input(batch_shape=(None,self.config.max_seq_size), name='input_layer')
        input_layer_2 = Input(batch_shape=(None,180), name='input_layer_cts')
        input_layer_3 = Input(batch_shape=(None,10), name='input_layer_feat')
        lda_input = Input(batch_shape=(None, 8), name='lda_input_layer')
        
        
        embed_op1 = Embedding(input_dim= self.config.word_vocab_size, input_length=90, output_dim=self.config.embedding_dim, 
                              weights=[embedding_matrix], trainable=True, name='word_embeddings')(input_layer)  
        spdropout1 = SpatialDropout1D(0.2, name='sp_dropout1')(embed_op1)
        conv1_word = Conv1D(filters=32, kernel_size=2, name='conv1_word')(spdropout1)
        maxpool1_word = GlobalMaxPool1D()(conv1_word)
        avgpool1_word = GlobalAvgPool1D()(conv1_word)
        
        conv2_word = Conv1D(filters=32, kernel_size=4, name='conv2_word')(spdropout1)
        maxpool2_word = GlobalMaxPool1D()(conv2_word)
        avgpool2_word = GlobalAvgPool1D()(conv2_word)
        
        conv3_word = Conv1D(filters=32, kernel_size=8, name='conv3_word')(spdropout1)
        maxpool3_word = GlobalMaxPool1D()(conv3_word)
        avgpool3_word = GlobalAvgPool1D()(conv3_word)
        
        embed_op_2 = Embedding(input_dim=len(t2.word_index)+1, input_length=180, output_dim=128, name='char_embeddings')(input_layer_2)
        spdropout_2 = SpatialDropout1D(0.2, name='sp_dropout_2')(embed_op_2)
        conv1 = Conv1D(filters=32, kernel_size=2, name='conv1')(spdropout_2)
        maxpool1 = GlobalMaxPool1D()(conv1)
        avgpool1 = GlobalAvgPool1D()(conv1)
        
        conv2 = Conv1D(filters=32, kernel_size=4, name='conv2')(spdropout_2)
        maxpool2 = GlobalMaxPool1D()(conv2)
        avgpool2 = GlobalAvgPool1D()(conv2)
        
        conv3 = Conv1D(filters=32, kernel_size=8, name='conv3')(spdropout_2)
        maxpool3 = GlobalMaxPool1D()(conv3)
        avgpool3 = GlobalAvgPool1D()(conv3)


        
        concat = concatenate([maxpool1_word,maxpool2_word, maxpool3_word,avgpool1_word, avgpool2_word, avgpool3_word,maxpool1,maxpool2, maxpool3,avgpool1, avgpool2, avgpool3,lda_input, input_layer_3], name='concatenate')
        
        dense1 = Dense(units=self.config.units3, activation='tanh', name='dense1')(concat)
#         drop1 = Dropout(0.1, name='Dropout_1_c3')(dense1)
        drop1 = GaussianDropout(0.1, name='Dropout_1_c3')(dense1)
        bn = BatchNormalization()(drop1)
        dense2 = Dense(units=16, activation='tanh', name='dense2')(bn)
        dense3 = Dense(units=1, activation='sigmoid', name='output')(dense2)
        classifier = keras.models.Model(inputs=[input_layer,input_layer_2,input_layer_3,lda_input], outputs=dense3)
        
        return classifier
config = Configuration.get_config_object()
model = Model(config)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
classifier = model.get_model()
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
classifier.summary()
earlystopping = EarlyStopping(monitor='val_f1', patience=1, mode='max')
modelcheckpoint = ModelCheckpoint(filepath='model.h5', monitor='val_f1', verbose=1, save_best_only=True, mode='max')
pbar = TQDMNotebookCallback(leave_inner=True, leave_outer=True)
DATA_SPLIT_SEED = 2018
clr = CyclicLR(base_lr=0.001, max_lr=0.002,
               step_size=300, mode='exp_range',
               gamma=0.99994)
callbacks1 = [earlystopping,modelcheckpoint, clr]
# cw = {0:0.84121908, 1: 1.56515283} val f1 = 0.6866
# cw = {0:1, 1: 1.6} val f1 = 0.6879
cw = {0:2, 1: 2.6}
classifier.fit([X_train_tts,X_train_cts,feat_train ,ldavec_train], y_train, callbacks=callbacks1 ,validation_data=([X_dev_tts,X_dev_cts,feat_dev,ldavec_dev], y_dev), epochs=1, batch_size=128, verbose=1, class_weight=cw)
cw = {0:1, 1: 1.6}
classifier.fit([X_train_tts,X_train_cts,feat_train ,ldavec_train], y_train, callbacks=callbacks1 ,validation_data=([X_dev_tts,X_dev_cts,feat_dev,ldavec_dev], y_dev), epochs=1, batch_size=256, verbose=1, class_weight=cw)
# pred_dev_c1 = classifier.predict([X_dev_tts,X_dev_cts,feat_dev,ldavec_dev], verbose=1)
# thresholds = []
# for thresh in np.arange(0.15, 0.601, 0.01):
# #     for alpha1 in np.arange(0.1, 0.81, 0.1):
# #         alpha2 = 1 - alpha1
#         thresh = np.round(thresh, 2)
# #         pred_dev_y = alpha1 * pred_dev_c1 + alpha2 * pred_dev_c2
#         res = f1_score(y_dev, (pred_dev_c1 > thresh).astype(int))
# #     thresholds.append([thresh, res, alpha1, alpha2])
#         thresholds.append([thresh, res])
#         print("F1 score at threshold {0}, is {1}".format(thresh, np.round(res,2)))
    
# thresholds.sort(key=lambda x: x[1], reverse=True)
# best_thresh = thresholds[0][0]
# best_f1 = thresholds[0][1]
# # best_alpha1 = thresholds[0][2]
# # best_alpha2 = thresholds[0][3]
# print("Best f1 score: {1} at threshold {0}".format(best_thresh, best_f1))
# pred1 = classifier.predict([test_tts,test_cts,feat_test,ldavec_test], verbose=1)
# pred2 = classifier2.predict([test_tts,test_cts,feat_test,ldavec_test], verbose=1)
# pred = best_alpha1*pred1 + best_alpha2*pred2
pred = classifier.predict([test_tts,test_cts,feat_test,ldavec_test], verbose=1)
pred = (pred > 0.5).astype(int)
test['prediction'] = pred
final = test.drop(['question_text','feat_vect','fp', 'sp', 'tps', 'tpp',  'conjunction', 'special_chars', 
                      'commas', 'qm', 'len',  'sm', 'clean','clean_2'], axis=1)
final.head()
final.to_csv('submission.csv',index=False)