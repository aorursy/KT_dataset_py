from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import numpy as np

import pandas as pd

import argparse

import collections

import datetime

import gc

import glob

import logging

import math

import operator

import os 

import pickle

import pkg_resources

import random

import re

import scipy.stats as stats

import seaborn as sns

import shutil

import sys

import time

import torch

import torch.nn as nn

import torch.utils.data

import torch.nn.functional as F

from contextlib import contextmanager

from collections import OrderedDict

# from nltk.stem import PorterStemmer

import scipy as sp

from sklearn import metrics

from sklearn import model_selection

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import check_is_fitted

from sklearn.feature_extraction.text import _document_frequency

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import make_pipeline, make_union

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from torch.nn import CrossEntropyLoss, MSELoss

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,

                              TensorDataset)

%load_ext autoreload

%autoreload 2

%matplotlib inline

# from tqdm import tqdm, tqdm_notebook, trange

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

tqdm.pandas()

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import warnings

warnings.filterwarnings('ignore')

# from apex import amp
!apt install curl mecab libmecab-dev mecab-ipadic-utf8 file -y

!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git

%cd mecab-ipadic-neologd

!bin/install-mecab-ipadic-neologd -n -a -y --prefix /var/lib/mecab/dic/mecab-ipadic-neologd

!sed -i -e 's@^dicdir.*$@dicdir = /var/lib/mecab/dic/mecab-ipadic-neologd@' /etc/mecabrc

!pip install mecab-python3

%cd ../../input/

!ls
import MeCab



class MecabTokenizer:

    def __init__(self):

        self.wakati = MeCab.Tagger('-Owakati')

        self.wakati.parse('')



    def tokenize(self, line):

        txt = self.wakati.parse(line)

        txt = txt.split()

        return txt

    

    def mecab_tokenizer(self, line):

        node = self.wakati.parseToNode(line)

        keywords = []

        while node:

            if node.feature.split(",")[0] == "名詞":

                keywords.append(node.surface)

            node = node.next

        return keywords       
tok = MecabTokenizer()

tok.mecab_tokenizer("kaggle days 楽しいイベントでしたね。") 
df = pd.DataFrame(columns=['id', 'text'])



df.id = ['kaeruhogehfuga0'+str(i) for i in range(10)]

df.text = [

    'BERT の事前学習タスク NSP と SOP の精度差を日本語の公開コーパスを用いて簡単に検証した。',

    'kaggle の discussion の upvote downvote 予測をしてみた',

    'PyTorch lightening で Titanic 問題解いてみた。',

    'モデルの蒸留を実装し freesound2019 コンペで検証してみた。',

    '単語 ID 列を長さでソートしてミニバッチ内で padding する。',

    'Kaggle Master になりました！',

    'PetFinder.my Adoption Prediction で準優勝しました！',

    'Google Colaboratory で fastText の pretrained model のSetup をする。',

    'PyTorch NN を用いて Titanic コンペに挑戦する。',

    '営業マンが1年でSEになって機械学習エンジニアに転職する話'

]



print(df.shape)

df.head()
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]





html_tags = ['<p>', '</p>', '<table>', '</table>', '<tr>', '</tr>', '<ul>', '<ol>', '<dl>', '</ul>', '</ol>',

             '</dl>', '<li>', '<dd>', '<dt>', '</li>', '</dd>', '</dt>', '<h1>', '</h1>',

             '<br>', '<br/>', '<strong>', '</strong>', '<span>', '</span>', '<blockquote>', '</blockquote>',

             '<pre>', '</pre>', '<div>', '</div>', '<h2>', '</h2>', '<h3>', '</h3>', '<h4>', '</h4>', '<h5>', '</h5>',

             '<h6>', '</h6>', '<blck>', '<pr>', '<code>', '<th>', '</th>', '<td>', '</td>', '<em>', '</em>']



empty_expressions = ['&lt;', '&gt;', '&amp;', '&nbsp;', 

                     '&emsp;', '&ndash;', '&mdash;', '&ensp;'

                     '&quot;', '&#39;']



other = ['span', 'style', 'href', 'input']





def pre_preprocess(x):

    return str(x).lower()



def rm_spaces(text):

    spaces = ['\u200b', '\u200e', '\u202a', '\u2009', '\u2028', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u3000', '\x10', '\x7f', '\x9d', '\xad',

              '\x97', '\x9c', '\x8b', '\x81', '\x80', '\x8c', '\x85', '\x92', '\x88', '\x8d', '\x80', '\x8e', '\x9a', '\x94', '\xa0', 

              '\x8f', '\x82', '\x8a', '\x93', '\x90', '\x83', '\x96', '\x9b', '\x9e', '\x99', '\x87', '\x84', '\x9f',

             ]

    for space in spaces:

            text = text.replace(space, ' ')

    return text



def remove_urls(x):

    x = re.sub(r'(https?://[a-zA-Z0-9.-]*)', r'', x)



    # original

    x = re.sub(r'(quote=\w+\s?\w+;?\w+)', r'', x)

    return x



def clean_html_tags(x, stop_words=[]):      

    for r in html_tags:

        x = x.replace(r, '')

    for r in empty_expressions:

        x = x.replace(r, ' ')

    for r in stop_words:

        x = x.replace(r, '')

    return x



def replace_num(text):

    text = re.sub('[0-9]{5,}', '', text)

    text = re.sub('[0-9]{4}', '', text)

    text = re.sub('[0-9]{3}', '', text)

    text = re.sub('[0-9]{2}', '', text)

    return text



def get_url_num(x):

    pattern = "https?://[\w/:%#\$&\?\(\)~\.=\+\-]+"

    urls = re.findall(pattern, x)

    return len(urls)





def clean_puncts(x):

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



#zenkaku = '０,１,２,３,４,５,６,７,８,９,（,）,＊,「,」,［,］,【,】,＜,＞,？,・,＃,＠,＄,％,＝'.split(',')

#hankaku = '0,1,2,3,4,5,6,7,8,9,q,a,z,w,s,x,c,d,e,r,f,v,b,g,t,y,h,n,m,j,u,i,k,l,o,p'.split(',')



def clean_text_jp(x):

    x = x.replace('。', '')

    x = x.replace('、', '')

    x = x.replace('\n', '') # 改行削除

    x = x.replace('\t', '') # タブ削除

    x = x.replace('\r', '')

    x = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', x) 

    x = re.sub(r'\[math\]', ' LaTex math ', x) # LaTex削除

    x = re.sub(r'\[\/math\]', ' LaTex math ', x) # LaTex削除

    x = re.sub(r'\\', ' LaTex ', x) # LaTex削除   

    #for r in zenkaku+hankaku:

    #    x = x.replace(str(r), '')

    x = re.sub(' +', ' ', x)

    return x





def preprocess(data):

    data = data.progress_apply(lambda x: pre_preprocess(x))

    data = data.progress_apply(lambda x: rm_spaces(x))

    data = data.progress_apply(lambda x: remove_urls(x))

    data = data.progress_apply(lambda x: clean_puncts(x))

    data = data.progress_apply(lambda x: replace_num(x))

    data = data.progress_apply(lambda x: clean_html_tags(x, stop_words=other))

    data = data.progress_apply(lambda x: clean_text_jp(x))

    return data
df['text'] = preprocess(df['text'])

df.head()
df['wakati_text'] = df['text'].progress_apply(lambda x: ' '.join(tok.mecab_tokenizer(x)))

df.head()
def get_sentence_features(train, col):

    train[col + '_num_chars'] = train[col].apply(len)

    train[col + '_num_words'] = train[col].apply(lambda x: len(x.split()))

    train[col + '_num_unique_words'] = train[col].apply(lambda comment: len(set(w for w in comment.split())))

    return train



df = get_sentence_features(df, 'wakati_text')

df.head()
n_components = 20



SEED = 1129



word_vectorizer = make_pipeline(

                TfidfVectorizer(sublinear_tf=True,

                                strip_accents='unicode',

                                analyzer='word',

                                token_pattern=r'\w{1,}',

                                stop_words='english',

                                ngram_range=(1, 2),

                                max_features=20000),

                make_union(

                    TruncatedSVD(n_components=n_components, random_state=SEED),

                    n_jobs=1,

                ),

             )



char_vectorizer = make_pipeline(

                TfidfVectorizer(sublinear_tf=True,

                                strip_accents='unicode',

                                analyzer='char',

                                stop_words='english',

                                ngram_range=(1, 4),

                                max_features=50000),

                make_union(

                    TruncatedSVD(n_components=n_components, random_state=SEED),

                    n_jobs=1,

                ),

             )



wakati_text_wd = word_vectorizer.fit_transform(df['wakati_text']).astype(np.float32)

wakati_text_ch = char_vectorizer.fit_transform(df['wakati_text']).astype(np.float32)



X = np.concatenate([wakati_text_wd, wakati_text_ch], axis=1)

X = pd.DataFrame(X, columns=['text_wd_tfidf_svd_{}'.format(i) for i in range(n_components)])

X.head()
df = pd.concat([df, X], axis=1)

print(df.shape)

df.head()
# fasttext



vocab = []



for text in tqdm(df['wakati_text']):

    for t in text.split():

        vocab.extend(tok.mecab_tokenizer(t))



vocab = list(set(vocab))

len(vocab)
vocab
path_w = '/kaggle/working/kaggle_vocab.txt'



with open(path_w, mode='w') as f:

    for l in tqdm(df['wakati_text'].values):

        f.write('\n')

        f.write(l)
!cat /kaggle/working/kaggle_vocab.txt
# 以下は local だとうまくいきます。 (権限がなくて !unzip ができなかった🤔)
# ! wget https://github.com/facebookresearch/fastText/archive/v0.9.1.zip

# ! unzip v0.9.1.zip -d /kaggle/working



# %cd /kaggle/working/fastText-0.9.1

# ! make



# !pip install fastText



# %cd ../../input
# !fastText-0.9.1/fasttext skipgram -input /kaggle/working/kaggle_vocab.txt -output /kaggle/working/ft_model_simple_50 -dim 50
# from gensim.models.wrappers.fasttext import FastText



# fasttextVectorizer = FastText.load_fasttext_format('drive/My Drive/anews_action_log/fasttext_model/kaggledays_model_body_150.bin')
# fasttextVectorizer.most_similar('営業')