!pip install rank_bm25
#from ktb import *

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer,PorterStemmer

from nltk.tokenize import word_tokenize

from sklearn.cluster import DBSCAN

from nltk.corpus import stopwords

from collections import  Counter

import matplotlib.pyplot as plt

import tensorflow as tf

import pyLDAvis.gensim

from tqdm.notebook import tqdm

import seaborn as sns

import pandas as pd

import numpy as np

import pyLDAvis

import gensim

import spacy

import os

from gensim.models import KeyedVectors,  keyedvectors

import multiprocessing as mp

import re

import os

import json

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

from annoy import AnnoyIndex

from functools import lru_cache 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path, PurePath

import pandas as pd

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

import nltk

from nltk.corpus import stopwords

nltk.download("punkt")

import re

from rank_bm25 import BM25Okapi
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict



import pandas as pd

import os

import pickle

import gzip

import numpy as np

import random as rn

from datetime import datetime



SEED = 777



output_dir = "../Results"



def write_submission(df, cols = None):

    if cols is None:

        cols = df.columns

    time_now = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")

    df[cols].to_csv(os.path.join(output_dir,f'submission-{time_now}.csv'), index=False, float_format='%.4f')

    

def read_object(file):

    with gzip.open(file, "rb") as f:

        return pickle.load(f)



def write_object(file, obj):

    with gzip.open(file, "wb") as f:

        pickle.dump(obj, f)



def cache_func(func,key):

    if not os.path.exists(f"cache"):

        os.mkdir("cache")

    key = key+func.__name__

    def inner_func(*args, **kwargs):

        try: 

            if os.path.exists(f"cache/{key}"):

                return read_object(f"cache/{key}")

        except:

            pass

        obj = func(*args, **kwargs)

        write_object(f"cache/{key}", obj)

        return obj

    return inner_func



def seed_everything(seed):

    rn.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



def reduce_mem_usage(df: pd.DataFrame,

                     verbose: bool = True) -> pd.DataFrame:

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if (c_min > np.iinfo(np.int8).min

                        and c_max < np.iinfo(np.int8).max):

                    df[col] = df[col].astype(np.int8)

                elif (c_min > np.iinfo(np.int16).min

                      and c_max < np.iinfo(np.int16).max):

                    df[col] = df[col].astype(np.int16)

                elif (c_min > np.iinfo(np.int32).min

                      and c_max < np.iinfo(np.int32).max):

                    df[col] = df[col].astype(np.int32)

                elif (c_min > np.iinfo(np.int64).min

                      and c_max < np.iinfo(np.int64).max):

                    df[col] = df[col].astype(np.int64)

            else:

                if (c_min > np.finfo(np.float16).min

                        and c_max < np.finfo(np.float16).max):

                    df[col] = df[col].astype(np.float16)

                elif (c_min > np.finfo(np.float32).min

                      and c_max < np.finfo(np.float32).max):

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'

    if verbose:

        print(msg)

    return df



def maddest(d : Union[np.array, pd.Series, List], axis : Optional[int]=None) -> np.array:  

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)



def batching(df : pd.DataFrame,

             batch_size : int,

             add_index : Optional[bool]=True) -> pd.DataFrame :



    df['batch_'+ str(batch_size)] = df.groupby(df.index//batch_size, sort=False)[df.columns[0]].agg(['ngroup']).values + 1

    df['batch_'+ str(batch_size)] = df['batch_'+ str(batch_size)].astype(np.uint16)

    if add_index:

        df['batch_' + str(batch_size) +'_idx'] = df.index  - (df['batch_'+ str(batch_size)] * batch_size)

        df['batch_' + str(batch_size) +'_idx'] = df['batch_' + str(batch_size) +'_idx'].astype(np.uint16)

    return df



def flattern_values(obj, func=None):

    res = []

    if isinstance(obj, dict):

        for v in obj.values:

            res.extend(flattern_values(v, func))

    elif isinstance(obj, list):

        for v in obj:

            res.extend(flattern_values(v, func))

    else:

        if func is not None:

            res.extend(flattern_values(func(obj), None))

        else:

            res.append(obj)



    return res





def apply2values(obj, func):

    res = None

    if isinstance(obj, dict):

        res = {k:apply2values(v, func) for k,v in obj.items}

    elif isinstance(obj, list):

        res = [apply2values(v, func) for v in obj]

    else:

        res = func(obj)

    return res



seed_everything(SEED)







from nltk.stem import WordNetLemmatizer,PorterStemmer

from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors,keyedvectors

import numpy as np

import re

from annoy import AnnoyIndex

from collections import defaultdict

import unidecode

from gensim.summarization.bm25 import BM25

from nltk.corpus import stopwords

from tqdm.notebook import tqdm



DEFAULT_REGEX = "[-,.\\/!@#$%^&*))_+=\(|\)|:|,|;|\.|’|”|“|\?|%|>|<]+"



stopwords_en = set(stopwords.words('english'))



def tokenize(text, regex=DEFAULT_REGEX):

    if text is None or (not isinstance(text,str) and np.isnan(text)):

        return []

    return [x for y in word_tokenize(str(text)) for x in re.split(regex, y)]



def analyze(tokens):

    tokens = [unidecode.unidecode(t).lower() for t in tokens 

        if len(t) > 1 and t not in stopwords_en and (not t.isnumeric() or t.isalpha())]

    return tokens

   



lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()



def lemmatize(word):

    return  lemmatizer.lemmatize(str(word))



def stem(word):

    return  lemmatizer.lemmatize(str(word))



def lemmatize_data(obj):

    return apply2values(obj, lemmatize)



def stem_data(obj):

    return apply2values(obj, stem)



def reduce_w3v_size(data, model_file, tokenization_regex = DEFAULT_REGEX, use_lemmatizer = True, use_stemmer = False):

    words_model = None

    data = set(flattern_values(data, lambda x: analyze(tokenize(x, tokenization_regex))))

    

    with open(model_file, 'rb') as fin:

        words_model = KeyedVectors.load_word2vec_format(fin, binary=True)

    def get_vec(word):

        if word in words_model:

            return words_model.word_vec(word) 

        if use_lemmatizer:

            word = lemmatize(word) 

            if word in words_model:

                return words_model.word_vec(word) 

        if use_stemmer:

            word = stem(word)

            if word in words_model:

                return words_model.word_vec(word) 

        return words_model.word_vec("unk")



    vecs = {w: get_vec(w) for w in data}

    m = keyedvectors.Word2VecKeyedVectors(vector_size=words_model.vector_size)

    m.add(list(vecs.keys()), list(vecs.values()))

    return m



def rank_items(scores, n_indexs, w = None, and_weight= 0.7):

    items = []

    for item, sc  in scores.items():

        sc = [w[k] * v if w is not None and k in w else v for k,v in sc.items()]

        score = min(sc) * (and_weight) + (1-and_weight) * (np.sum(sc) + (n_indexs -len(sc)) * max(sc))/n_indexs

        items.append((item, score))

    items = sorted(items, key = lambda x: x[1])

    return items

        





class AnnoySearch:

    def __init__(self, w2v_model, columns, use_lemmatizer = True, use_stemmer= False):

        self.words_model = w2v_model

        self.use_lemmatizer = use_lemmatizer

        self.use_stemmer = use_stemmer

        self.index = {c:AnnoyIndex(w2v_model.vector_size, 'dot') for c in columns}

        

    def get_vector(self, text):

        def get_vec(word):

            if word in self.words_model:

                return self.words_model.word_vec(word) 

            if self.use_lemmatizer:

                word = lemmatize(word) 

                if word in self.words_model:

                    return self.words_model.word_vec(word) 

            if self.use_stemmer:

                word = stem(word)

                if word in self.words_model:

                    return self.words_model.word_vec(word) 

            return self.words_model.word_vec("unk")

        

        text = analyze(tokenize(text))

        if len(text) == 0:

            text = ["unk"]

        vector = np.mean([get_vec(w) for w in text] ,axis=0)

        return vector

    

    def build(self, df, n_trees=1000):

        self.ids_index = df.index.to_list()

        self.ids = dict(zip(self.ids_index, range(len(df))))

        self.df = df

        for i, row in tqdm(df.iterrows(), total = len(df)):

            for c, idx in self.index.items():

                idx.add_item(self.ids[i], self.get_vector(row[c]))

        for c, idx in self.index.items():

            idx.build(n_trees)

        

    def save(self):

        write_object("annoy_search.gz", self)

        

    def load(self):

        newObj =  read_object("annoy_search.gz")

        self.__dict__.update(newObj.__dict__)

    

    def query(self, q, n_items, w = None, and_weight= 0.7, include_distances= False):

        q_vec = self.get_vector(q)

        score_dict = defaultdict(dict)

        for c, idx in self.index.items():

            ids =  idx.get_nns_by_vector(q_vec, n_items, include_distances= True)

            for i, s in list(zip(list(ids[0]),list(ids[1]))):

                score_dict[i].update({c:s})

        sorted_res = rank_items(score_dict, len(self.index), w, and_weight)

        

        if not include_distances:

            sorted_res = [k for k,v in sorted_res]

        return sorted_res



        

class HybridSearch:

    def __init__(self, w2v_model, columns, use_lemmatizer = True, use_stemmer= True, use_lemmatizer_annoy = True, use_stemmer_annoy= False):

        self.use_lemmatizer = use_lemmatizer

        self.use_stemmer = use_stemmer

        self.annoy_search = AnnoySearch(w2v_model, columns, use_lemmatizer_annoy, use_lemmatizer_annoy)

        self.bm25 = {c:None for c in columns}



    def process_text(self, text):

        text = analyze(tokenize(text))

        if self.use_lemmatizer:

            text = lemmatize_data(text)

        if self.use_stemmer:

            text = stem_data(text)

        return text



    def build(self, df, n_trees=1000):

        for c in tqdm(self.bm25.keys()):

            self.bm25[c] = BM25(df[c].transform(self.process_text).to_list())

        self.annoy_search.build(df, n_trees)

    

    def save(self):

        write_object("hybrid_search.gz", self)

        

    def load(self):

        newObj =  read_object("hybrid_search.gz")

        self.__dict__.update(newObj.__dict__) 

    

    def query(self, q, n_items, w = None, and_weight= 0.7, include_distances=False, additive_semantic_weight = 0.1):

        score_dict = defaultdict(dict)

        q_tokens = self.process_text(q)

        for c, idx in self.bm25.items():

            scores =  idx.get_scores(q_tokens)

            for i, s in zip(range(len(scores)), scores):

                score_dict[i].update({c: -s})

        sorted_res = rank_items(score_dict, len(self.bm25), w, and_weight)

        bm25_dict = dict(sorted_res)

        annoy_res = self.annoy_search.query(q, n_items, w, and_weight, True)

        for item,score in annoy_res:

             bm25_dict[item] = score*additive_semantic_weight + bm25_dict[item]*score

        sorted_res = sorted(bm25_dict.items(), key = lambda x: x[1])

        results = self.annoy_search.df.iloc[[idx for idx,_ in sorted_res]][list(self.bm25.keys())]

        if include_distances:

            results["Score"] = [score for _,score in sorted_res]

        return results[:n_items]
data_path = "/kaggle/input/CORD-19-research-challenge"

all_sources = pd.read_csv(os.path.join(data_path,"metadata.csv"), dtype={'Microsoft Academic Paper ID': str,

                                      'pubmed_id': str})

csv_data_path = "/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv"

all_text_data = []

for file in os.listdir(csv_data_path):

    if os.path.splitext(file)[-1] == ".csv":

        print(f"loading {file}")

        all_text_data.append(pd.read_csv(os.path.join(csv_data_path, file)))

all_text_data = pd.concat(all_text_data)

duplicate_paper = ~(all_sources.title.isnull() | all_sources.abstract.isnull()) & (all_sources.duplicated(subset=['title', 'abstract']))

all_sources = all_sources[~duplicate_paper].reset_index(drop=True)

all_data = pd.merge(all_sources[list(set(all_sources.columns).difference(set(all_text_data.columns)))], all_text_data, how='left', left_on=['sha'], right_on = ['paper_id'])

print(f"all_sources shape {all_sources.shape}")

print(f"all_text_data shape {all_text_data.shape}")

print(f"all_data shape {all_data.shape}")
all_data.head()
search_columns = ["title", "abstract", "text", 'title', 'abstract', 'doi', 'authors', 'journal']
w2v_model = KeyedVectors.load("/kaggle/input/covid19-bio-w2v/small_w2v_model.wv")
#Hyperd model cant work on kaggle kernel because of memroy limit if you want to run it run it localy

#search = HybridSearch(w2v_model,columns=["title", "abstract", "text"])



search = AnnoySearch(w2v_model,columns=search_columns)

%timeit

search.build(all_data)
from ipywidgets import interact

import ipywidgets as widgets

import pandas as pd



def set_column_width(ColumnWidth, MaxRows):

    pd.options.display.max_colwidth = ColumnWidth

    pd.options.display.max_rows = MaxRows

    print('Set pandas dataframe column width to', ColumnWidth, 'and max rows to', MaxRows)

    

interact(set_column_width, 

         ColumnWidth=widgets.IntSlider(min=50, max=400, step=50, value=200),

         MaxRows=widgets.IntSlider(min=50, max=500, step=100, value=100));
from IPython.display import display



def search_papers(SearchTerms: str):

    search_results = search.query(SearchTerms, n_items=100, include_distances=True)

    results = all_data.iloc[[x for x,_ in search_results]]

    results["score"] = [x for _,x in search_results]

    if len(results) > 0:

        display(results) 

    return results



searchbar = widgets.interactive(search_papers, SearchTerms='viruse')

searchbar