# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.text import *

from fastai.tabular import *



from pathlib import Path

from typing import *



import torch

import torch.optim as optim



import gc

gc.collect()



import re

import os

import re

import gc

import pickle  

import random

import keras



import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import keras.backend as K



from keras.models import Model

from keras.layers import Dense, Input, Dropout, Lambda

from keras.optimizers import Adam

from keras.callbacks import Callback

from scipy.stats import spearmanr, rankdata

from os.path import join as path_join

from numpy.random import seed

from urllib.parse import urlparse

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold, train_test_split

from sklearn.linear_model import LogisticRegression

from bayes_opt import BayesianOptimization

from lightgbm import LGBMRegressor

from nltk.tokenize import wordpunct_tokenize

from nltk.stem.snowball import EnglishStemmer

from nltk.stem import WordNetLemmatizer

from functools import lru_cache

from tqdm import tqdm as tqdm
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
train.shape, test.shape, sub.shape
train.head()
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"couldnt" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"doesnt" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"havent" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"shouldnt" : "should not",

"that's" : "that is",

"thats" : "that is",

"there's" : "there is",

"theres" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"theyre":  "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"}





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





def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





def replace_typical_misspell(text):

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)





def clean_data(df, columns: list):

    for col in columns:

        df[col] = df[col].apply(lambda x: clean_numbers(x))

        df[col] = df[col].apply(lambda x: clean_text(x.lower()))

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))



    return df
target_cols_questions = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written']



target_cols_answers = ['answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']



targets = target_cols_questions + target_cols_answers
train = clean_data(train, ['answer', 'question_body', 'question_title'])

test = clean_data(test, ['answer', 'question_body', 'question_title'])
find = re.compile(r"^[^.]*")



train['netloc_1'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_1'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



train['netloc_2'] = train['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_2'] = test['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



train['netloc_3'] = train['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_3'] = test['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

train.head(3)
train.host.value_counts()
train.netloc_1.value_counts()
train.shape, test.shape
# train_tfidf = train.copy()

# test_tfidf = test.copy()
# import gensim



# w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../input/word2vec-google/GoogleNews-vectors-negative300.bin', 

#                                                             binary=True, unicode_errors='ignore')
# #https://www.kaggle.com/sediment/a-gentle-introduction-eda-tfidf-word2vec/data#Benchmark-with-Word2Vec



# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



# TFIDF_SVD_WORDVEC_DIM = 300



# def get_text_feats(df, col):



#     def tokenize_downcase_filtering(x):

#         words = TOKENIZER.tokenize(x)

#         lower_case = map(lambda w: w.lower(), words)

#         content_words = filter(lambda w: w not in STOPWORDS, lower_case)

#         return ' '.join(content_words)



#     rows = df[col].map(tokenize_downcase_filtering).values.tolist()

#     tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))  # dont use sklearn default tokenization tool 

#     tfidf_weights = tfidf.fit_transform(rows)

#     svd = TruncatedSVD(n_components=TFIDF_SVD_WORDVEC_DIM, n_iter=10)  # reduce dimensionality

#     dense_tfidf_repr_mat = svd.fit_transform(tfidf_weights)

    

#     word2vec_repr_mat = np.zeros((len(df), w2v_model.vector_size))

#     for i, row in enumerate(rows):

#         word2vec_accum = np.zeros((w2v_model.vector_size, ))

#         word_cnt = 0

#         for w in row.split(' '):

#             if w in w2v_model.wv:

#                 word2vec_accum += w2v_model.wv[w]

#                 word_cnt += 1



#         # compute the average for the wordvec of each non-sptop word

#         if word_cnt != 0:

#             word2vec_repr_mat[i] = word2vec_accum / word_cnt



#     return  np.concatenate([word2vec_repr_mat, dense_tfidf_repr_mat], axis=1)  # word2vec + tfidf
# from nltk.tokenize import RegexpTokenizer

# from nltk.corpus import stopwords



# TOKENIZER = RegexpTokenizer(r'\w+')

# STOPWORDS = set(stopwords.words('english'))
# # let's build features

# df_all = pd.concat((train_tfidf, test_tfidf))

# df_all['question_title_len'] = df_all['question_title'].map(lambda x: len(TOKENIZER.tokenize(x)))

# df_all['question_body_len'] = df_all['question_body'].map(lambda x: len(TOKENIZER.tokenize(x)))

# df_all['answer_len'] = df_all['answer'].map(lambda x: len(TOKENIZER.tokenize(x)))
# text_cols = [

#     'question_title',

#     'question_body',

#     'answer'

# ]



# text_len_cols = ['question_title_len', 'question_body_len', 'answer_len']
# from sklearn.decomposition import TruncatedSVD



# data = []

# for col in text_cols:

#     data.append(get_text_feats(df_all, col))



# data.append(df_all[text_len_cols].values)

# data = np.concatenate(data, axis=1)



# train_feats = data[:len(train_tfidf)]

# test_feats = data[len(train_tfidf):]





# print(train_feats.shape)
# print(test_feats.shape)
# del w2v_model

# gc.collect()
# train_wordvec = pd.DataFrame(data = train_feats)

# train_wordvec.columns = [str(col) + '_col' for col in train_wordvec.columns]

# print(train_wordvec.shape)

# train_wordvec.head()
# test_wordvec = pd.DataFrame(data = test_feats)

# test_wordvec.columns = [str(col) + '_col' for col in test_wordvec.columns]

# print(test_wordvec.shape)

# test_wordvec.head()
# tabular_cols = ['question_user_name', 'answer_user_name', 

#                'netloc_1', 'netloc_2', 'netloc_3',

#                'category', 'host']



# train_select = train[tabular_cols + targets]

# test_select = test[tabular_cols]



# train_tfidf_final = pd.concat([train_wordvec, train_select], axis=1)



# test_tfidf_final = pd.concat([test_wordvec, test_select], axis=1)



# gc.collect()
# valid_sz = 2000

# valid_idx = range(len(train_wordvec)-valid_sz, len(train_wordvec))

# valid_idx
# cont_names = train_wordvec.columns

# cat_names = tabular_cols

# dep_var = targets

# procs = [FillMissing, Categorify, Normalize]



# test_tab = TabularList.from_df(test_tfidf_final, cat_names=cat_names, cont_names=cont_names, procs=procs)



# data = (TabularList.from_df(train_tfidf_final, procs = procs, cont_names=cont_names, cat_names=cat_names)

#         .split_by_idx(valid_idx)

#         .label_from_df(cols=dep_var)

#         .add_test(test_tab)

#         .databunch(bs=32))
# from fastai.callbacks import *



# auroc = AUROC()



# learn_wordvec = tabular_learner(data, layers=[800, 400, 200, 100], 

#                         ps=[0.5, 0.5, 0.25, 0.25], emb_drop=0.5)

# learn_wordvec.lr_find()

# learn_wordvec.recorder.plot(suggestion=True)
# lr = 5e-2

# learn_wordvec.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 0.75)
# learn_wordvec.lr_find()

# learn_wordvec.recorder.plot(suggestion=True)
# lr = 1e-4

# learn_wordvec.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 0.75)
# learn_wordvec.lr_find()

# learn_wordvec.recorder.plot(suggestion=True)
# lr = 1e-5

# learn_wordvec.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 1.)
# pred_test_wordvec, lbl_test_wordvec = learn_wordvec.get_preds(ds_type=DatasetType.Test)

# print(pred_test_wordvec.shape)

# pred_test_wordvec
# pred_test_wordvec = np.clip(pred_test_wordvec, 0.00001, 0.999999)

# pred_test_wordvec.shape
# del df_all, train_tfidf_final, train_tfidf, test_tfidf, test_tfidf_final, train_wordvec, test_wordvec

# gc.collect()
train_tfidf = train.copy()

test_tfidf = test.copy()
stemmer = EnglishStemmer()



@lru_cache(30000)

def stem_word(text):

    return stemmer.stem(text)





lemmatizer = WordNetLemmatizer()



@lru_cache(30000)

def lemmatize_word(text):

    return lemmatizer.lemmatize(text)





def reduce_text(conversion, text):

    return " ".join(map(conversion, wordpunct_tokenize(text.lower())))





def reduce_texts(conversion, texts):

    return [reduce_text(conversion, str(text))

            for text in tqdm(texts)]
train_tfidf['question_body'] = reduce_texts(stem_word, train_tfidf['question_body'])

test_tfidf['question_body'] = reduce_texts(stem_word, test_tfidf['question_body'])



train_tfidf['question_title'] = reduce_texts(stem_word, train_tfidf['question_title'])

test_tfidf['question_title'] = reduce_texts(stem_word, test_tfidf['question_title'])



train_tfidf['answer'] = reduce_texts(stem_word, train_tfidf['answer'])

test_tfidf['answer'] = reduce_texts(stem_word, test_tfidf['answer'])
train_text_1 = train_tfidf['question_body']

test_text_1 = test_tfidf['question_body']

all_text_1 = pd.concat([train_text_1, test_text_1])



train_text_2 = train_tfidf['answer']

test_text_2 = test_tfidf['answer']

all_text_2 = pd.concat([train_text_2, test_text_2])



train_text_3 = train_tfidf['question_title']

test_text_3 = test_tfidf['question_title']

all_text_3 = pd.concat([train_text_3, test_text_3])
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import scipy

from sklearn.metrics import log_loss

import xgboost as xgb

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.sparse import hstack

from sklearn.decomposition import TruncatedSVD





word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=80000,

    tokenizer=tokenize)

word_vectorizer.fit(all_text_1)



train_word_features_1 = word_vectorizer.transform(train_text_1)

test_word_features_1 = word_vectorizer.transform(test_text_1)



word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=80000,

    tokenizer=tokenize)

word_vectorizer.fit(all_text_2)



train_word_features_2 = word_vectorizer.transform(train_text_2)

test_word_features_2 = word_vectorizer.transform(test_text_2)



word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=80000,

    tokenizer=tokenize)

word_vectorizer.fit(all_text_3)



train_word_features_3 = word_vectorizer.transform(train_text_3)

test_word_features_3 = word_vectorizer.transform(test_text_3)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    token_pattern=r'\w{1,}',

    ngram_range=(1, 4),

    max_features=50000,

    tokenizer=tokenize)

char_vectorizer.fit(all_text_1)



train_char_features_1 = char_vectorizer.transform(train_text_1)

test_char_features_1 = char_vectorizer.transform(test_text_1)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    token_pattern=r'\w{1,}',

    ngram_range=(1, 4),

    max_features=50000,

    tokenizer=tokenize)

char_vectorizer.fit(all_text_2)



train_char_features_2 = char_vectorizer.transform(train_text_2)

test_char_features_2 = char_vectorizer.transform(test_text_2)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(1, 4),

    max_features=50000,

    tokenizer=tokenize)

char_vectorizer.fit(all_text_3)



train_char_features_3 = char_vectorizer.transform(train_text_3)

test_char_features_3 = char_vectorizer.transform(test_text_3)



train_features = hstack([train_char_features_1, train_word_features_1, train_char_features_2, train_word_features_2,train_char_features_3, train_word_features_3])

test_features = hstack([test_char_features_1, test_word_features_1, test_char_features_2, test_word_features_2,test_char_features_3, test_word_features_3])



pca = TruncatedSVD(n_components=300, n_iter=10)

tf_idf_text_train = pca.fit_transform(train_features)

tf_idf_text_test = pca.fit_transform(test_features)
train_tfidf = pd.DataFrame(data = tf_idf_text_train)

train_tfidf.columns = [str(col) + '_col' for col in train_tfidf.columns]

print(train_tfidf.shape)

train_tfidf.head()
test_tfidf = pd.DataFrame(data = tf_idf_text_test)

test_tfidf.columns = [str(col) + '_col' for col in test_tfidf.columns]

print(test_tfidf.shape)

test_tfidf.head()
tabular_cols = ['question_user_name', 'answer_user_name', 

               'netloc_1', 'netloc_2', 'netloc_3',

               'category', 'host']



train_select = train[tabular_cols + targets]

test_select = test[tabular_cols]



train_tfidf_final = pd.concat([train_tfidf, train_select], axis=1)



test_tfidf_final = pd.concat([test_tfidf, test_select], axis=1)



gc.collect()
valid_sz = 2000

valid_idx = range(len(train_tfidf)-valid_sz, len(train_tfidf))

valid_idx
train_tfidf.columns
cont_names = train_tfidf.columns

cat_names = tabular_cols

dep_var = targets

procs = [FillMissing, Categorify, Normalize]



test_tab = TabularList.from_df(test_tfidf_final, cat_names=cat_names, cont_names=cont_names, procs=procs)



data = (TabularList.from_df(train_tfidf_final, procs = procs, cont_names=cont_names, cat_names=cat_names)

        .split_by_idx(valid_idx)

        .label_from_df(cols=dep_var)

        .add_test(test_tab)

        .databunch(bs=32))
from fastai.callbacks import *



auroc = AUROC()



learn_tfidf = tabular_learner(data, layers=[800, 400, 200, 100], 

                        ps=[0.5, 0.5, 0.25, 0.25], emb_drop=0.5)

learn_tfidf.lr_find()

learn_tfidf.recorder.plot(suggestion=True)
lr = 5e-2

learn_tfidf.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 0.75)
learn_tfidf.lr_find()

learn_tfidf.recorder.plot(suggestion=True)
lr=1e-4

learn_tfidf.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 1)
learn_tfidf.lr_find()

learn_tfidf.recorder.plot(suggestion=True)
lr=1e-5

learn_tfidf.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 1.)
pred_test_tfidf, lbl_test_tfidf = learn_tfidf.get_preds(ds_type=DatasetType.Test)
pred_test_tfidf = np.clip(pred_test_tfidf, 0.00001, 0.999999)

pred_test_tfidf.shape
pred_test_tfidf
X, y  = train_tfidf_final.iloc[:, :-30], train_tfidf_final.iloc[:, -30:]
y.head()
# Categorical boolean mask

categorical_feature_mask = X.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = X.columns[categorical_feature_mask].tolist()



# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()



# apply le on categorical feature columns

X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

X[categorical_cols].head(10)
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.multioutput import MultiOutputRegressor





regr_multirf = MultiOutputRegressor(LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=5, learning_rate=0.1, 

                                                  n_estimators=100, min_child_samples=20, subsample=0.8, 

                                                  subsample_freq=0, colsample_bytree=0.8, 

                                                  reg_alpha=1., reg_lambda=1., random_state=42, silent=False))



regr_multirf.fit(X, y)
X = test_tfidf_final

X.head()
# Categorical boolean mask

categorical_feature_mask = X.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = X.columns[categorical_feature_mask].tolist()



# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()



# apply le on categorical feature columns

X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

X[categorical_cols].head(10)
pred_test_tree = regr_multirf.predict(X)
pred_test_tree = np.clip(pred_test_tree, 0.00001, 0.999999)

pred_test_tree.shape
pred_test_tree
train, val = train_test_split(train, test_size=0.2, shuffle=True)
train.shape, val.shape
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

text = train.question_title.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

text = train.question_body.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

text = train.answer.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# who asked the most questions



qn_asker = train.question_user_name.value_counts()

qn_asker
import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.style as style

style.use('seaborn-poster')

style.use('ggplot')
qn_asker.loc[qn_asker>10].sort_values().plot(kind = 'barh', figsize=(15,15)).legend(loc='best')
qn_answerer = train.answer_user_name.value_counts()

qn_answerer
qn_answerer.loc[qn_answerer>10].sort_values().plot(kind = 'barh', figsize=(15,15)).legend(loc='best')
category = train.category.value_counts()

category
## lets see some distributions of questions targets

plt.figure(figsize=(20, 5))



sns.distplot(train[target_cols_questions[0]], hist= False , rug= False ,kde=True, label =target_cols_questions[0],axlabel =False )

sns.distplot(train[target_cols_questions[1]], hist= False , kde=True, rug= False,label =target_cols_questions[1],axlabel =False)

sns.distplot(train[target_cols_questions[2]], hist= False , kde=True, rug= False,label =target_cols_questions[2],axlabel =False)

sns.distplot(train[target_cols_questions[3]], hist= False , kde=True, rug= False,label =target_cols_questions[3],axlabel =False)

sns.distplot(train[target_cols_questions[4]], hist= False , kde=True, rug= False,label =target_cols_questions[4],axlabel =False)

plt.show()
## lets see some distributions of questions targets

plt.figure(figsize=(20, 5))



sns.distplot(train[target_cols_answers[0]], hist= False , rug= False ,kde=True, label =target_cols_answers[0],axlabel =False )

sns.distplot(train[target_cols_answers[1]], hist= False , kde=True, rug= False,label =target_cols_answers[1],axlabel =False)

sns.distplot(train[target_cols_answers[2]], hist= False , kde=True, rug= False,label =target_cols_answers[2],axlabel =False)

sns.distplot(train[target_cols_answers[3]], hist= False , kde=True, rug= False,label =target_cols_answers[3],axlabel =False)

sns.distplot(train[target_cols_answers[4]], hist= False , kde=True, rug= False,label =target_cols_answers[4],axlabel =False)

plt.show()
bs, bptt = 32, 80



data_lm = TextLMDataBunch.from_df('.', train, val, test,

                  include_bos=False,

                  include_eos=False,

                  text_cols=['question_title', 'question_body', 'answer'],

                  label_cols=targets,

                  bs=bs,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



data_lm.save('data_lm.pkl')
# src_lm = ItemLists(path, TextList.from_df(train, path=".", cols = [ 'question_title', "question_body", 'answer']), 

#                    TextList.from_df(val, path=".", cols = [ 'question_title', "question_body", 'answer']))
# data_lm = src_lm.label_for_lm().databunch(bs=32)
path = "."

data_lm = load_data(path, 'data_lm.pkl', bs=bs, bptt=bptt)
path = "."

data_bwd = load_data(path, 'data_lm.pkl', bs=bs, bptt = bptt, backwards=True)
data_lm.show_batch()
data_bwd.show_batch()
awd_lstm_lm_config = dict( emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,

                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)
awd_lstm_clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.4,

                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5,

                               config=awd_lstm_lm_config, pretrained = False)

learn = learn.to_fp16(clip=0.1)
fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn.load_pretrained(*fnames, strict=False)

learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(5e-3, 5e-2), moms=(0.8, 0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3))
learn.save('fit_head')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr = slice(1e-4, 5e-2), moms=(0.8, 0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4,  1e-2))
learn.recorder.plot_losses()
learn.save('fine-tuned')

learn.load('fine-tuned')

learn.save_encoder('fine-tuned-fwd')
print(learn.model[0].encoder)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(learn.model[0].encoder.weight.data)

embedding_weights = pca.transform(learn.model[0].encoder.weight.data)
plt.figure(figsize=(15,15))

plt.scatter(embedding_weights[:, 0], embedding_weights[:, 1])



for i, word in enumerate(data_lm.vocab.itos[:50]):

    plt.annotate(word, xy=(embedding_weights[i, 0], embedding_weights[i, 1]))

plt.show()
learn = language_model_learner(data_bwd, AWD_LSTM, drop_mult=0.5,

                               config=awd_lstm_lm_config, pretrained = False)

learn = learn.to_fp16(clip=0.1)
fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn.load_pretrained(*fnames, strict=False)

learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(5e-2, 1e-1), moms=(0.8, 0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3))
learn.save('fit_head-bwd')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr = slice(1e-4, 1e-3), moms=(0.8, 0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4,  1e-2))
learn.recorder.plot_losses()
learn.save('fine-tuned-bwd')

learn.load('fine-tuned-bwd')

learn.save_encoder('fine-tuned-bwd')
text_cols = ['question_title', "question_body", 'answer']
data_cls = TextClasDataBunch.from_df('.', train, val, test, vocab = data_lm.vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols=text_cols,

                  label_cols=targets,

                  bs=bs,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



data_cls.save('data_cls.pkl')
data_cls = load_data(path, 'data_cls.pkl', bs=bs)
data_cls.show_batch()
data_cls_bwd = load_data(path, 'data_cls.pkl', bs=bs, backwards=True)
data_cls_bwd.show_batch()
class L1LossFlat(nn.MSELoss):

    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:

        return super().forward(input.view(-1), target.view(-1))
learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.5,config=awd_lstm_clas_config, pretrained = False)

learn.load_encoder('fine-tuned-fwd')

learn = learn.to_fp16(clip=0.1)

#learn.loss_func = L1LossFlat()

fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn.load_pretrained(*fnames, strict=False)

learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(1e-2, 1e-1), moms=(0.8, 0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn.save('first-head')

learn.load('first-head')
learn.freeze_to(-2)

learn.fit_one_cycle(2, slice(1e-1/(2.6**4),1e-1), moms=(0.8,0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn.save('second')

learn.load('second')
learn.freeze_to(-3)

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn.save('third')

learn.load('third')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1e-5/(2.6**4),1e-4), moms=(0.8,0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn.recorder.plot_losses()
learn.save('fwd-cls')
learn_bwd = text_classifier_learner(data_cls_bwd, AWD_LSTM, drop_mult=0.5, config=awd_lstm_clas_config, pretrained = False)

learn_bwd.load_encoder('fine-tuned-bwd')

learn_bwd = learn_bwd.to_fp16(clip=0.1)
fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn_bwd.load_pretrained(*fnames, strict=False)

learn_bwd.freeze()
learn_bwd.lr_find()

learn_bwd.recorder.plot(suggestion=True)
learn_bwd.fit_one_cycle(2, max_lr=slice(5e-2, 1e-1), moms=(0.8, 0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn_bwd.save('first-head-bwd')

learn_bwd.load('first-head-bwd')
learn_bwd.freeze_to(-2)

learn_bwd.fit_one_cycle(2, slice(1e-1/(2.6**4),1e-1), moms=(0.8,0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn_bwd.save('second-bwd')

learn_bwd.load('second-bwd')
learn_bwd.freeze_to(-3)

learn_bwd.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn_bwd.save('third-bwd')

learn_bwd.load('third-bwd')
learn_bwd.unfreeze()

learn_bwd.lr_find()

learn_bwd.recorder.plot(suggestion=True)
learn_bwd.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7), pct_start=0.3, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))
learn_bwd.recorder.plot_losses()
learn_bwd.save('bwd-cls')
pred_fwd_val, lbl_fwd_val = learn.get_preds(ds_type=DatasetType.Valid,ordered=True)

pred_bwd_val, lbl_bwd_val = learn_bwd.get_preds(ds_type=DatasetType.Valid,ordered=True)
pred_fwd_test, lbl_fwd_test = learn.get_preds(ds_type=DatasetType.Test,ordered=True)

pred_bwd_test, lbl_bwd_test = learn_bwd.get_preds(ds_type=DatasetType.Test,ordered=True)
pred_test_tree = torch.from_numpy(pred_test_tree)

final_preds_test = (0.30 * pred_fwd_test + 0.30 * pred_bwd_test + 0.30 * pred_test_tfidf  + 0.10* pred_test_tree)
# def get_ordered_preds(learn, ds_type, preds):

#   np.random.seed(42)

#   sampler = [i for i in learn.data.dl(ds_type).sampler]

#   reverse_sampler = np.argsort(sampler)

#   preds = [p[reverse_sampler] for p in preds]

#   return preds
# val_raw_preds = learn.get_preds(ds_type=DatasetType.Valid)

# val_preds_fwd = get_ordered_preds(learn, DatasetType.Valid, val_raw_preds)



# val_raw_preds = learn_bwd.get_preds(ds_type=DatasetType.Valid)

# val_preds_bwd = get_ordered_preds(learn_bwd, DatasetType.Valid, val_raw_preds)
# final_preds = (pred_fwd + pred_bwd)/2
# from scipy.stats import spearmanr

# score = 0

# for i in range(30):

#     score += np.nan_to_num(spearmanr(val[targets].values[:, i], final_preds_val[:, i]).correlation) / 30

# score
# test_raw_preds = learn.get_preds(ds_type=DatasetType.Test)

# test_preds_fwd = get_ordered_preds(learn, DatasetType.Test, test_raw_preds)



# test_raw_preds = learn_bwd.get_preds(ds_type=DatasetType.Test)

# test_preds_bwd = get_ordered_preds(learn_bwd, DatasetType.Test, test_raw_preds)
sub.head()



sub.iloc[:, 1:] = final_preds_test.numpy()

sub.to_csv('submission.csv', index=False)

sub.head()
fig, axes = plt.subplots(6, 5, figsize=(18, 15))

axes = axes.ravel()

bins = np.linspace(0, 1, 20)



for i, col in enumerate(targets):

    ax = axes[i]

    sns.distplot(train[col], label=col, bins=bins, ax=ax, color='blue')

    sns.distplot(sub[col], label=col, bins=bins, ax=ax, color='orange')

    # ax.set_title(col)

    ax.set_xlim([0, 1])

plt.tight_layout()

plt.show()

plt.close()