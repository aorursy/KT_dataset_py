import numpy as np, pandas as pd

import json

import ast 

from textblob import TextBlob

import nltk

import torch

import pickle

from scipy import spatial

import warnings

warnings.filterwarnings('ignore')

import spacy

from nltk import Tree

en_nlp = spacy.load('en')

from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
# !conda update pandas --y
train = pd.read_csv("../input/sentence-embedding/train.csv",encoding="latin-1")
train.shape
with open("../input/sent-embedding/dict_embeddings1.pickle", "rb") as f:

    d1 = pickle.load(f)
with open("../input/sent-embedding/dict_embeddings2.pickle", "rb") as f:

    d2 = pickle.load(f)
dict_emb = dict(d1)

dict_emb.update(d2)
len(dict_emb)
del d1, d2
def get_target(x):

    idx = -1

    for i in range(len(x["sentences"])):

        if x["text"] in x["sentences"][i]: idx = i

    return idx
train.head(3)
train.shape
train.dropna(inplace=True)
train.shape
def process_data(train):

    

    print("step 1")

    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])

    

    print("step 2")

    train["target"] = train.apply(get_target, axis = 1)

    

    print("step 3")

    train['sent_emb'] = train['sentences'].apply(lambda x: [dict_emb[item][0] if item in\

                                                           dict_emb else np.zeros(4096) for item in x])

    print("step 4")

    train['quest_emb'] = train['question'].apply(lambda x: dict_emb[x] if x in dict_emb else np.zeros(4096) )

        

    return train   
train = process_data(train)
train.head(3)
train.to_csv("procesed_train.csv")
del train
import gc

gc.collect()