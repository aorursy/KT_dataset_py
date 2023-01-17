!pip install -q pandarallel

!pip install -q spacy 

!pip install -q spacy_cld

!pip install -q pyspellchecker

!python -m spacy download xx_ent_wiki_sm > /dev/null
import numpy as np

import pandas as pd



import os

import gc



import spacy

from spacy_cld import LanguageDetector

import xx_ent_wiki_sm



from spellchecker import SpellChecker



import matplotlib.pyplot as plt

import seaborn as sns



import time

import random

from tqdm.notebook import tqdm

tqdm.pandas()



import re

import nltk



from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
nlp = xx_ent_wiki_sm.load()

language_detector = LanguageDetector()

nlp.add_pipe(language_detector)
def get_lang_score(text, lang):

    try:

        doc = nlp(str(text))

        language_scores = doc._.language_scores

        return language_scores.get(lang, 0)

    except Exception:

        return 0
# Loading data



train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train1['lang'] = 'en'



train_es = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')

train_es['lang'] = 'es'



train_fr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-fr-cleaned.csv')

train_fr['lang'] = 'fr'



train_pt = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-pt-cleaned.csv')

train_pt['lang'] = 'pt'



train_ru = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-ru-cleaned.csv')

train_ru['lang'] = 'ru'



train_it = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-it-cleaned.csv')

train_it['lang'] = 'it'



train_tr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv')

train_tr['lang'] = 'tr'



train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)

train2['lang'] = 'en'



train = pd.concat([

    

    train1[['comment_text', 'lang', 'toxic']],

    train_es[['comment_text', 'lang', 'toxic']],

    train_tr[['comment_text', 'lang', 'toxic']],

    train_fr[['comment_text', 'lang', 'toxic']],

    train_pt[['comment_text', 'lang', 'toxic']],

    train_ru[['comment_text', 'lang', 'toxic']],

    train_it[['comment_text', 'lang', 'toxic']],

    train2[['comment_text', 'lang', 'toxic']]

    

]).sample(n=20000).reset_index(drop=True)



del train1, train_es, train_fr, train_pt, train_ru, train_it, train_tr, train2

gc.collect()
train['lang_score'] = train.progress_apply(lambda x: get_lang_score(x['comment_text'], x['lang']), axis=1)
sns.distplot(train['lang_score'])
train = train[train['lang_score'] > 0.8]
spell = SpellChecker()



# A quick example

misspelled = spell.unknown(['something', 'somegting', 'helo', 'fack', 'here', 'bijour'])
# Counting the number of spelling errors



train['mispell_count'] = train['comment_text'].progress_apply(lambda x: len(spell.unknown(x.split())))
train[train['mispell_count'] < 100]['mispell_count'].hist(bins=100)
sns.countplot(train['lang'])
sns.countplot(train[train['mispell_count'] < 20]['lang'])
train = train[train['mispell_count'] < 20]
train['comment_text'] = train['comment_text'].apply(lambda x: x.replace('\n', ' '))
train.head()