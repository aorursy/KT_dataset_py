import pickle

import pandas as pd



def read_competision_file(train=True):

    if train:

        with open('../input/pickled-jigsaw-unintended-bias-in-toxicity-train/train.pickle','rb') as f:

            df = pickle.load(f)

    else:

        df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

    return df
"""

Created on Fri May 31 14:20:00 2019

@author: Toshiyuki Sakamoto

"""

import re

import os

import json

import bz2

import numpy as np

from tqdm import tqdm

from fastcache import clru_cache as lru_cache

from textblob import Word

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import TweetTokenizer, RegexpTokenizer

from multiprocessing import Pool



def trans_text(s, dict):

    return s.translate(dict)

tknzr = TweetTokenizer(strip_handles=True)

reg_twt = re.compile(r"(https:\/\/twitter[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")

reg_wik = re.compile(r"(https:\/\/[a-z][a-z]\.wikipedia\.[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")

reg_utb1 = re.compile(r"(https:\/\/youtu\.be[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")

reg_utb2 = re.compile(r"(https:\/\/www\.youtube\.com[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")

reg_url1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")

reg_url2 = re.compile(r"(bit\.ly\/[a-zA-Z0-9]+)")

reg_url3 = re.compile(r"(goo\.gl\/[a-zA-Z0-9]+)")

reg_url4 = re.compile(r"(www\.[a-zA-Z][a-zA-Z0-9\.]*[a-zA-Z0-9]+\.(com|org|net|int|edu|gov|mil|([a-z][a-z])))")

reg_tid = re.compile(r"(@[a-zA-Z0-9]+)")

reg_tag = re.compile(r"(#[a-zA-Z0-9]+)")

reg_num = re.compile(r"([0-9][0-9]+)")

prefix_word_list = ['trans','super','hyper','every','cyber','anti','bull','and','any','pre','non','sub','car','dis','mis','neo','at','un','ex','be','co','re','de','by']

emoji_isolate_dict = {'(8':'🙂', '(:':'🙂', '(:>':'🙂', '(;':'🙂', '(=':'🙂', '(=>':'🙂', ')\';':'🙁', ')8':'🙂', '):':'🙁', ');':'🙁', ')=':'🙁', ':\'(':'🙂', ':\')':'🙂', ':\'D':'🙂', ':\'d':'😜', ':(':'🙁', ':)':'🙂', ':*(':'🙁', ':*)':'🙂', ':8':'🙂', ':;':'🙁', ':=':'🙂', ':D':'🙂',':O(':'🙂', ':O)':'🙂', ':Op':'😜', ':P':'😜', ':[':'🙁', ':\'':'🙁', ':]':'🙂', ':d':'😜', ':o':'🙂', ':o(':'🙁', ':o)':'🙂', ':oD':'🙂', ':o\'':'🙁', ':op':'😜', ':p':'😜', ':|':'🙁', ':}':'🙂', ';\')':'🙂', ';(':'🙁', ';)':'🙂', ';:':'🙁', ';D':'🙂', ';O)':'🙂', ';P':'😜', ';[':'🙁', ';]':'🙂', ';d':'😜', ';o(':'🙁', ';o)':'🙂', ';oD':'🙂', ';oP':'😜', ';od':'😜', ';op':'😜', ';o}':'🙂', ';p':'😜', ';{':'🙁', ';}':'🙂', '=(':'🙁', '=)':'🙂', '=:':'🙁', '=D':'🙂', '=Op':'😜', '=P':'😜', '=\'':'🙁', '=d':'😜', '=o)':'🙂', '=op':'🙁', '=o}':'🙂', '=p':'😜', '=}':'🙂', '>:':'🙁', '>:(':'🙁', '>:)':'🙁', '>;':'🙂', '>=p':'😜', '@:':'🙂', '@;':'🙁', '{:':'🙂', '{;':'🙂', '{=':'🙂', '|:':'🙁', '}:':'🙁', '};':'🙁'}

separatedict = {ord(k):' ' for k in "!\"\'-#$%&()*+/:,.;=@[\\]^_`{|}~\t\r\n"}

def replace_text(s, deletedict, isolatedict):

    s = re.sub(reg_twt, ' <twt> ' ,s)

    s = re.sub(reg_wik, ' <wik> ' ,s)

    s = re.sub(reg_utb1, ' <utb> ' ,s)

    s = re.sub(reg_utb2, ' <utb> ' ,s)

    s = re.sub(reg_url1, ' <url> ' ,s)

    s = re.sub(reg_url2, ' <url> ' ,s)

    s = re.sub(reg_url3, ' <url> ' ,s)

    s = re.sub(reg_url4, ' <url> ' ,s)

    s = re.sub(reg_tid, ' <tid> ' ,s)

    s = re.sub(reg_tag, ' <tag> ' ,s)

    s = s.replace('?', ' <qes> ')

    s = s.replace('\n', ' <ret> ')

    s = re.sub(reg_num, ' <num> ' ,s)

    s = s.translate(deletedict)

    return s.translate(isolatedict)



def getsimbols(s):

    all_charactors = set()

    for cc in str(s):

        if ord(cc) >= 256:

            all_charactors.add(cc)

    return all_charactors



stemmer = PorterStemmer()

@lru_cache(120000)

def stem(s):

    return stemmer.stem(s)

@lru_cache(120000)

def corr(s):

    return str(Word(s).correct())

def get_til(s, dict_words):

    if s in dict_words:

        return s

    s_t = s.title()

    if s_t in dict_words:

        return s_t

    s_l = s.lower()

    if s_l in dict_words:

        return s_l

    return None

def get_stil(s, dict_words):

    t = get_til(s, dict_words)

    if t is not None:

        return t

    s = stem(s)

    t = get_til(s, dict_words)

    if t is not None:

        return t

def get_heads(s, pref, dict_words):

    p = pref + '-' + s[len(pref):]

    t = get_stil(p, dict_words)

    if t is not None:

        return [t]

    p = pref + '-' + s[len(pref):].lower()

    t = get_stil(p, dict_words)

    if t is not None:

        return [t]

    p = pref + '-' + s[len(pref):].title()

    t = get_stil(p, dict_words)

    if t is not None:

        return [t]

    p = pref + '-' + stem(s[len(pref):])

    t = get_til(p, dict_words)

    if t is not None:

        return [t]

    p = pref + '-' + stem(s[len(pref):].lower())

    t = get_til(p, dict_words)

    if t is not None:

        return [t]

    p = pref + '-' + stem(s[len(pref):].title())

    t = get_til(p, dict_words)

    if t is not None:

        return [t]

    t1 = get_til(pref, dict_words)

    t2 = get_stil(s[len(pref):], dict_words)

    if t1 is not None and t2 is not None:

        return [t1,t2]

    return None

def get_stem(s, dict_words, spell_collector):

    t = get_stil(s, dict_words)

    if t is not None:

        return [t]

    if '-' in s:

        t = get_stil(s.replace('-',''), dict_words)

        if t is not None:

            return [t]

    if '\'' in s:

        t = get_stil(s.replace('\'',''), dict_words)

        if t is not None:

            return [t]

    for pref in prefix_word_list:

        if len(s) > len(pref)+1 and s.lower().startswith(pref):

            tl = get_heads(s, pref, dict_words)

            if tl is not None:

                return tl

    if type(spell_collector) == dict:

        if s in spell_collector:

            p = spell_collector[s]

            t = get_stil(p, dict_words)

            if t is not None:

                return [t]

    elif str(type(spell_collector)) == "<class 'function'>":

        p = spell_collector(s)

        t = get_stil(p, dict_words)

        if t is not None:

            return [t]

    elif type(spell_collector) == str and spell_collector == 'textblob':

        p = corr(s)

        t = get_stil(p, dict_words)

        if t is not None:

            return [t]

    return None

def fix_spell(s, dict_words, spell_collector):

    if s in emoji_isolate_dict:

        return [emoji_isolate_dict[s]]

    tl = get_stem(s, dict_words, spell_collector)

    if tl is not None:

        return tl

    t = s.translate(separatedict)

    if t != s:

        v = [get_stem(u, dict_words, spell_collector) for u in t.split()]

        return sum([w for w in v if w is not None],[])

    return [s]

def tokenize_text(s, dict_words, spell_collector):

    return sum([fix_spell(w, dict_words, spell_collector) for w in tknzr.tokenize(s)],[])



class TokenVectorizer:



    def __init__(self, vector_dict, uniform_unknown_word=False, lemma_dict={}, spell_collector='textblob', num_process=-1):

        self.vector_dict = vector_dict

        self.uniform_unknown_word = uniform_unknown_word

        self.lemma_dict = lemma_dict

        self.spell_collector = spell_collector

        self.num_process = num_process if num_process>0 else os.cpu_count()

        self.word_index = dict()

        self.unknown_words = set()

        if type(self.vector_dict) == str:

            self.load_vector_dict(self.vector_dict)

        if type(self.lemma_dict) == str:

            if os.path.isfile(self.lemma_dict):

                with open(self.lemma_dict) as f:

                    self.lemma_dict = json.load(f)

            else:

                self.lemma_dict = dict()

        if type(self.spell_collector) == str:

            if os.path.isfile(self.spell_collector):

                with open(self.spell_collector) as f:

                    self.spell_collector = json.load(f)

        self.deletedict = dict()

        self.isolatedict = dict()



    def load_vector_dict(self, vector_file):

        if vector_file.endswith('.pickle') or vector_file.endswith('.pkl'):

            with open(vector_file, 'rb') as f:

                self.vector_dict = pickle.load(f)

        elif vector_file.endswith('.bz2'):

            with bz2.BZ2File(vector_file, 'rb') as f:

                self.vector_dict = pickle.loads(f.read())

        else:

            def get_coefs(word, *arr):

                return word, np.asarray(arr, dtype='float32')

            with open(vector_file) as f:

                self.vector_dict = dict(get_coefs(*line.strip().split(' ')) for line in f)

        if '<url>' not in self.vector_dict:

            v = self.vector_dict.values()

            mat_size = np.max([len(t) for t in v])

            emb_mean, emb_std = np.mean([np.mean(t) for t in v]), np.mean([np.std(t) for t in v])

            np.random.seed(12)

            self.vector_dict['<tid>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<tag>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<qes>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<ret>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<twt>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<wik>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<utb>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<url>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

            self.vector_dict['<num>'] = np.random.normal(emb_mean, emb_std, (mat_size,))

    def preprocess(self, string_list):

        lenmadict = {ord(k):v for k,v in self.lemma_dict.items()}

        all_simbols = set()

        dict_words = set(self.vector_dict.keys())

        if self.num_process <= 1:

            if len(lenmadict) > 0:

                string_list = [s.translate(lenmadict) for s in string_list]

            for s in string_list:

                all_simbols |= getsimbols(s)

        else:

            if len(lenmadict) > 0:

                with Pool(self.num_process) as pl:

                    r = pl.starmap(trans_text, [ (s,lenmadict) for s in string_list ], chunksize=10000)

                    for i,p in enumerate(r):

                        string_list[i] = p

            with Pool(self.num_process) as pl:

                r = pl.imap_unordered(getsimbols, string_list, chunksize=10000)

                for p in r:

                    all_simbols |= p

        self.deletedict = {ord(k):'' for k in (all_simbols - dict_words)}

        self.isolatedict = {ord(k):' %s '%k for k in (dict_words & all_simbols)}

        if self.num_process <= 1:

            for i,s in enumerate(string_list):

                string_list[i] = replace_text(s,self.deletedict,self.isolatedict)

        else:

            with Pool(self.num_process) as pl:

                r = pl.starmap(replace_text, [ (s,self.deletedict,self.isolatedict) for s in string_list ], chunksize=10000)

                for i,p in enumerate(r):

                    string_list[i] = p

        return string_list





    def tokenize(self, string_list, maxlen=-1, pad_sequence=True):

        dict_words = set(self.vector_dict.keys())



        all_seq = []

        for s in string_list:

            all_seq.append([])

        if self.num_process <= 1:

            for i,s in enumerate(string_list):

                all_seq[i] = tokenize_text(s, dict_words, self.spell_collector)

        else:

            with Pool(self.num_process) as pl:

                r = pl.starmap(tokenize_text, [ (s, dict_words, self.spell_collector) for s in string_list ], chunksize=10000)

                for i,t in enumerate(r):

                    all_seq[i] = t



        for i,s_token in enumerate(all_seq):

            tokens = []

            for t in s_token:

                if t in self.word_index:

                    tokens.append(self.word_index[t])

                else:

                    self.word_index[t] = len(self.word_index) + 1

                    tokens.append(self.word_index[t])

            all_seq[i] = tokens



        if pad_sequence:

            if maxlen <= 0:

                maxlen = np.max([len(t) for t in all_seq])

            for i,tokens in enumerate(all_seq):

                if len(tokens) > maxlen:

                    all_seq[i] = tokens[len(tokens)-maxlen:]

                elif len(tokens) < maxlen:

                    all_seq[i] = [0 for w in range(maxlen-len(tokens))] + tokens

            return np.array(all_seq, dtype=np.int64)

        else:

            if maxlen > 0:

                for i,tokens in enumerate(all_seq):

                    if len(tokens) > maxlen:

                        all_seq[i] = tokens[len(tokens)-maxlen:]

            return all_seq



    def vectorize(self):

        v = self.vector_dict.values()

        mat_size = np.max([len(t) for t in v])

        if self.uniform_unknown_word:

            emb_mean, emb_std = np.mean([np.mean(t) for t in v]), np.mean([np.std(t) for t in v])

        embedding_matrix = np.zeros((len(self.word_index) + 1, mat_size))

        for word_o, i in self.word_index.items():

            if word_o in self.vector_dict:

                embedding_matrix[i] = self.vector_dict[word_o]

            else:

                if self.uniform_unknown_word:

                    embedding_matrix[i] = np.random.normal(self.emb_mean, self.emb_std, (mat_size,))

                self.unknown_words.add(word_o)

        return embedding_matrix



    def __call__(self, string_list, maxlen=-1, pad_sequence=True):

        print("Preprocess.")

        string_list = self.preprocess(string_list)

        print("Tokenize.")

        tokens = self.tokenize(string_list,maxlen,pad_sequence)

        print("Vectorize.")

        return tokens, self.vectorize()
import sys

sys.path.append("../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT")
def load_bert_vocab_as_embed(vocab_file='../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'):

    """Loads a vocabulary file into a dictionary."""

    vocab = dict()

    index = 0

    with open(vocab_file, "r", encoding="utf-8") as reader:

        while True:

            token = reader.readline()

            if not token:

                break

            token = token.strip()

            vocab[token] = [index]

            index += 1

    return vocab
def convert_lines(example, max_seq_length, tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    return np.array(all_tokens)
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertForSequenceClassification

def get_bert_tokens():

    tokenizer = BertTokenizer.from_pretrained('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/', cache_dir=None,do_lower_case=True)

    test_df = read_competision_file(False)

    test_df['comment_text'] = test_df['comment_text'].astype(str) 

    return convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), 380, tokenizer)

"""

def get_bert_tokens():

    vocab_emb = load_bert_vocab_as_embed()

    lemma_simple = {"\u00c1": "A", "\u1d8f": "a", "\u00e1": "a", "\u00c0": "A", "\u00e0": "a", "\u00c2": "A", "\u00e2": "a", "\u01cd": "A", "\u01ce": "a", "\u0102": "A", "\u0103": "a", "\u00c3": "A", "\u00e3": "a", "\u1ea2": "A", "\u1ea3": "a", "\u0226": "A", "\u0227": "a", "\u1ea0": "A", "\u1ea1": "a", "\u00c4": "A", "\u00e4": "a", "\u00c5": "A", "\u00e5": "a", "\u1e00": "A", "\u1e01": "a", "\u0100": "A", "\u0101": "a", "\u0104": "A", "\u0105": "a", "\u023a": "A", "\u2c65": "a", "\u0200": "A", "\u0201": "a", "\u1ea4": "A", "\u1ea5": "a", "\u1ea6": "A", "\u1ea7": "a", "\u1eaa": "A", "\u1eab": "a", "\u1ea8": "A", "\u1ea9": "a", "\u1eac": "A", "\u1ead": "a", "\u1eae": "A", "\u1eaf": "a", "\u1eb0": "A", "\u1eb1": "a", "\u1eb4": "A", "\u1eb5": "a", "\u1eb2": "A", "\u1eb3": "a", "\u1eb6": "A", "\u1eb7": "a", "\u01fa": "A", "\u01fb": "a", "\u01e0": "A", "\u01e1": "a", "\u01de": "A", "\u01df": "a", "\u0202": "A", "\u0203": "a", "\u2c6d": "A", "\u0251": "a", "\uff21": "A", "\uff41": "a", "\u1e02": "B", "\u1e03": "b", "\u1e04": "B", "\u1e05": "b", "\u1e06": "B", "\u1e07": "b", "\u0243": "B", "\u0180": "b", "\u0181": "B", "\u0253": "b", "\u0182": "B", "\u0183": "b", "\u1d6c": "B", "\u1d80": "b", "\u0299": "B", "\uff22": "b", "\uff42": "B", "\u0238": "d", "\u2114": "B", "\u0106": "C", "\u0107": "c", "\u0108": "C", "\u0109": "c", "\u010c": "C", "\u010d": "c", "\u010a": "C", "\u010b": "c", "\u0304": "c", "\u00c7": "C", "\u00e7": "c", "\u1e08": "C", "\u1e09": "c", "\u023b": "C", "\u023c": "c", "\u0187": "C", "\u0188": "c", "\u0255": "C", "\u1d04": "c", "\uff23": "C", "\uff43": "c", "\u010e": "D", "\u010f": "d", "\u1e0a": "D", "\u1e0b": "d", "\u1e10": "D", "\u1e11": "d", "\u1e0c": "D", "\u1e0d": "d", "\u1e12": "D", "\u1e13": "d", "\u1e0e": "D", "\u1e0f": "d", "\u0110": "D", "\u0111": "d", "\u0326": "d", "\u0189": "D", "\u0256": "d", "\u018a": "D", "\u0257": "d", "\u018b": "D", "\u018c": "d", "\u1d6d": "D", "\u1d81": "d", "\u1d91": "D", "\u0221": "d", "\u1d05": "D", "\uff24": "d", "\uff44": "D", "\u00c9": "E", "\u00e9": "e", "\u00c8": "E", "\u00e8": "e", "\u00ca": "E", "\u00ea": "e", "\u1e18": "E", "\u1e19": "e", "\u011a": "E", "\u011b": "e", "\u0114": "E", "\u0115": "e", "\u1ebc": "E", "\u1ebd": "e", "\u1e1a": "E", "\u1e1b": "e", "\u1eba": "E", "\u1ebb": "e", "\u0116": "E", "\u0117": "e", "\u00cb": "E", "\u00eb": "e", "\u0112": "E", "\u0113": "e", "\u0228": "E", "\u0229": "e", "\u0118": "E", "\u0119": "e", "\u1d92": "E", "\u0246": "e", "\u0247": "E", "\u0204": "e", "\u0205": "E", "\u1ebe": "e", "\u1ebf": "E", "\u1ec0": "e", "\u1ec1": "E", "\u1ec4": "e", "\u1ec5": "E", "\u1ec2": "e", "\u1ec3": "E", "\u1e1c": "e", "\u1e1d": "E", "\u1e16": "e", "\u1e17": "E", "\u1e14": "e", "\u1e15": "E", "\u0206": "e", "\u0207": "E", "\u1eb8": "e", "\u1eb9": "E", "\u1ec6": "e", "\u1ec7": "E", "\u2c78": "e", "\u1d07": "E", "\uff25": "e", "\uff45": "E", "\u1e1e": "F", "\u1e1f": "f", "\u0191": "F", "\u0192": "f", "\u1d6e": "F", "\u1d82": "f", "\ua730": "F", "\uff26": "f", "\uff46": "F", "\u01f4": "G", "\u01f5": "g", "\u011e": "G", "\u011f": "g", "\u011c": "G", "\u011d": "g", "\u01e6": "G", "\u01e7": "g", "\u0120": "G", "\u0121": "g", "\u0122": "G", "\u0123": "g", "\u1e20": "G", "\u1e21": "g", "\u01e4": "G", "\u01e5": "g", "\u0193": "G", "\u0260": "g", "\u1d83": "G", "\u0262": "g", "\uff27": "G", "\uff47": "g", "\u0124": "H", "\u0125": "h", "\u021e": "H", "\u021f": "h", "\u1e26": "H", "\u1e27": "h", "\u1e22": "H", "\u1e23": "h", "\u1e28": "H", "\u1e29": "h", "\u1e24": "H", "\u1e25": "h", "\u1e2a": "H", "\u1e2b": "h", "\u1e96": "h", "\u0126": "H", "\u0127": "h", "\u2c67": "H", "\u2c68": "h", "\u029c": "H", "\u0266": "h", "\uff28": "H", "\uff48": "h", "\u00cd": "I", "\u00ed": "i", "\u00cc": "I", "\u00ec": "i", "\u012c": "I", "\u012d": "i", "\u00ce": "I", "\u00ee": "i", "\u01cf": "I", "\u01d0": "i", "\u00cf": "I", "\u00ef": "i", "\u1e2e": "I", "\u1e2f": "i", "\u0128": "I", "\u0129": "i", "\u012e": "I", "\u012f": "i", "\u012a": "I", "\u012b": "i", "\u1ec8": "I", "\u1ec9": "i", "\u0208": "I", "\u0209": "i", "\u020a": "I", "\u020b": "i", "\u1eca": "I", "\u1ecb": "i", "\u1e2c": "I", "\u1e2d": "i", "\u0197": "I", "\u0268": "i", "\u1d7b": "I", "\u0130": "I", "\u1d96": "i", "\u026a": "I", "\u0131": "i", "\uff29": "I", "\uff49": "i", "\u0134": "J", "\u0135": "j", "\u0248": "J", "\u0249": "j", "\u01f0": "j", "\u0237": "J", "\u03f3": "j", "\u029d": "J", "\u025f": "j", "\u0284": "J", "\u1d0a": "j", "\uff2a": "J", "\uff4a": "j", "\u1e30": "K", "\u1e31": "k", "\u01e8": "K", "\u01e9": "k", "\u0136": "K", "\u0137": "k", "\u1e32": "K", "\u1e33": "k", "\u1e34": "K", "\u1e35": "k", "\u0198": "K", "\u0199": "k", "\u2c69": "K", "\u2c6a": "k", "\u1d84": "k", "\ua740": "K", "\ua741": "k", "\u1d0b": "K", "\uff2b": "k", "\uff4b": "K", "\u0139": "L", "\u013a": "l", "\u013d": "L", "\u013e": "l", "\u013b": "L", "\u013c": "l", "\u1e36": "L", "\u1e37": "l", "\u1e38": "L", "\u1e39": "l", "\u1e3c": "L", "\u1e3d": "l", "\u1e3a": "L", "\u1e3b": "l", "\u0141": "L", "\u0142": "l", "\u013f": "L", "\u0140": "l", "\u023d": "L", "\u019a": "l", "\u2c60": "L", "\u2c61": "l", "\u2c62": "L", "\u026b": "l", "\u026c": "L", "\u1d85": "l", "\u026d": "L", "\u0234": "l", "\u029f": "L", "\uff2c": "l", "\uff4c": "L", "\u1e3e": "M", "\u1e3f": "m", "\u1e40": "M", "\u1e41": "m", "\u1e42": "M", "\u1e43": "m", "\u1d6f": "M", "\u1d86": "m", "\u2c6e": "M", "\u0271": "m", "\u1d0d": "M", "\uff2d": "m", "\uff4d": "M", "\u0143": "N", "\u0144": "n", "\u01f8": "N", "\u01f9": "n", "\u0147": "N", "\u0148": "n", "\u00d1": "N", "\u00f1": "n", "\u1e44": "N", "\u1e45": "n", "\u0145": "N", "\u0146": "n", "\u1e46": "N", "\u1e47": "n", "\u1e4a": "N", "\u1e4b": "n", "\u1e48": "N", "\u1e49": "n", "\u019d": "N", "\u0272": "n", "\u0220": "N", "\u019e": "n", "\u1d70": "N", "\u1d87": "n", "\u0274": "N", "\u0273": "n", "\uff2e": "N", "\uff4e": "n", "\u014a": "N", "\u014b": "n", "\u0235": "n", "\u00d3": "O", "\u00f3": "o", "\u00d2": "O", "\u00f2": "o", "\u014e": "O", "\u014f": "o", "\u00d4": "O", "\u00f4": "o", "\u1ed0": "O", "\u1ed1": "o", "\u1ed2": "O", "\u1ed3": "o", "\u1ed6": "O", "\u1ed7": "o", "\u1ed4": "O", "\u1ed5": "o", "\u01d1": "O", "\u01d2": "o", "\u00d6": "O", "\u00f6": "o", "\u022a": "O", "\u022b": "o", "\u0150": "O", "\u0151": "o", "\u00d5": "O", "\u00f5": "o", "\u1e4c": "O", "\u1e4d": "o", "\u1e4e": "O", "\u1e4f": "o", "\u022c": "O", "\u022d": "o", "\u022e": "O", "\u022f": "o", "\u0230": "O", "\u0231": "o", "\u00d8": "O", "\u00f8": "o", "\u01fe": "O", "\u01ff": "o", "\u01ea": "O", "\u01eb": "o", "\u01ec": "O", "\u01ed": "o", "\u014c": "O", "\u014d": "O", "\u1e52": "O", "\u1e53": "o", "\u0302": "o", "\u1e50": "O", "\u1e51": "o", "\u1ece": "O", "\u1ecf": "o", "\u020c": "O", "\u020d": "o", "\u020e": "O", "\u020f": "o", "\u01a0": "O", "\u01a1": "o", "\u1eda": "O", "\u1edb": "o", "\u1edc": "O", "\u1edd": "o", "\u1ee0": "O", "\u1ee1": "o", "\u1ede": "O", "\u1edf": "o", "\u1ee2": "O", "\u1ee3": "o", "\u1ecc": "O", "\u1ecd": "o", "\u1ed8": "O", "\u1ed9": "o", "\u019f": "O", "\u0275": "o", "\u2c7a": "O", "\u1d0f": "o", "\uff2f": "O", "\uff4f": "o", "\u1e54": "P", "\u1e55": "p", "\u1e56": "P", "\u1e57": "p", "\u2c63": "P", "\u1d7d": "p", "\u01a4": "P", "\u01a5": "p", "\u1d71": "P", "\u1d88": "p", "\u1d18": "P", "\uff30": "p", "\uff50": "P", "\u024a": "Q", "\u024b": "q", "\uff31": "Q", "\uff51": "q", "\u0239": "Q", "\u02a0": "q", "\u0154": "R", "\u0155": "r", "\u0158": "R", "\u0159": "r", "\u1e58": "R", "\u1e59": "r", "\u0156": "R", "\u0157": "r", "\u0210": "R", "\u0211": "r", "\u0212": "R", "\u0213": "r", "\u1e5a": "R", "\u1e5b": "r", "\u1e5c": "R", "\u1e5d": "r", "\u1e5e": "R", "\u1e5f": "r", "\u024c": "R", "\u024d": "r", "\u2c64": "R", "\u027d": "r", "\u1d72": "R", "\u1d89": "r", "\u027c": "R", "\u027e": "r", "\u1d73": "R", "\u0280": "r", "\uff32": "R", "\uff52": "r", "\u015a": "S", "\u015b": "s", "\u1e64": "S", "\u1e65": "s", "\u015c": "S", "\u015d": "s", "\u0160": "S", "\u0161": "s", "\u1e66": "S", "\u1e67": "s", "\u1e60": "S", "\u1e61": "s", "\u015e": "S", "\u015f": "s", "\u1e62": "S", "\u1e63": "s", "\u1e68": "S", "\u1e69": "s", "\u0218": "S", "\u0219": "s", "\u1d74": "s", "\u1d8a": "s", "\u0282": "s", "\u023f": "s", "\ua731": "s", "\uff33": "S", "\uff53": "s", "\u0164": "T", "\u0165": "t", "\u1e6a": "T", "\u1e6b": "t", "\u0162": "T", "\u0163": "t", "\u1e6c": "T", "\u1e6d": "t", "\u021a": "T", "\u021b": "t", "\u1e70": "T", "\u1e71": "t", "\u1e6e": "T", "\u1e6f": "t", "\u0166": "T", "\u0167": "t", "\u023e": "T", "\u2c66": "t", "\u01ac": "T", "\u01ad": "t", "\u01ae": "T", "\u0288": "t", "\u1e97": "t", "\u1d75": "T", "\u01ab": "t", "\u0236": "T", "\uff34": "t", "\u1d1b": "T", "\uff54": "T", "\u00da": "U", "\u00fa": "u", "\u00d9": "U", "\u00f9": "u", "\u016c": "U", "\u016d": "u", "\u00db": "U", "\u00fb": "u", "\u01d3": "U", "\u01d4": "u", "\u016e": "U", "\u016f": "u", "\u00dc": "U", "\u00fc": "u", "\u01d7": "U", "\u01d8": "u", "\u01db": "U", "\u01dc": "u", "\u01d9": "U", "\u01da": "u", "\u01d5": "U", "\u01d6": "u", "\u0170": "U", "\u0171": "u", "\u0168": "U", "\u0169": "u", "\u1e78": "U", "\u1e79": "u", "\u0172": "U", "\u0173": "u", "\u016a": "U", "\u016b": "u", "\u1e7a": "U", "\u1e7b": "u", "\u1ee6": "U", "\u1ee7": "u", "\u0214": "U", "\u0215": "u", "\u0216": "U", "\u0217": "u", "\u01af": "U", "\u01b0": "u", "\u1ee8": "U", "\u1ee9": "u", "\u1eea": "U", "\u1eeb": "u", "\u1eee": "U", "\u1eef": "u", "\u1eec": "U", "\u1eed": "u", "\u1ef0": "U", "\u1ef1": "u", "\u1ee4": "U", "\u1ee5": "u", "\u1e72": "U", "\u1e73": "u", "\u1e76": "U", "\u1e77": "u", "\u1e74": "U", "\u1e75": "u", "\u0244": "U", "\u0289": "u", "\u1d7e": "U", "\u1d99": "u", "\u1d1c": "U", "\uff35": "u", "\uff55": "U", "\u1e7c": "V", "\u1e7d": "v", "\u1e7e": "V", "\u1e7f": "v", "\u01b2": "V", "\u028b": "v", "\u1d8c": "v", "\u2c71": "V", "\u2c74": "v", "\uff36": "V", "\u1d20": "v", "\uff56": "v", "\u1e82": "W", "\u1e83": "w", "\u1e80": "W", "\u1e81": "w", "\u0174": "W", "\u0175": "w", "\u1e84": "W", "\u1e85": "w", "\u1e86": "W", "\u1e87": "w", "\u1e88": "W", "\u1e89": "w", "\u1e98": "w", "\u2c72": "W", "\u2c73": "w", "\uff37": "W", "\uff57": "w", "\u1e8c": "X", "\u1e8d": "x", "\u1e8a": "X", "\u1e8b": "x", "\u1d8d": "X", "\uff38": "x", "\uff58": "X", "\u00dd": "Y", "\u00fd": "y", "\u1ef2": "Y", "\u1ef3": "y", "\u0176": "Y", "\u0177": "y", "\u0178": "Y", "\u1e99": "y", "\u00ff": "y", "\u1ef8": "Y", "\u1ef9": "y", "\u1e8e": "Y", "\u1e8f": "y", "\u0232": "Y", "\u0233": "y", "\u1ef6": "Y", "\u1ef7": "y", "\u1ef4": "Y", "\u1ef5": "y", "\u024e": "Y", "\u024f": "y", "\u01b3": "Y", "\u01b4": "y", "\uff39": "Y", "\u028f": "y", "\uff59": "y", "\u0179": "Z", "\u017a": "z", "\u1e90": "Z", "\u1e91": "z", "\u017d": "Z", "\u017e": "z", "\u017b": "Z", "\u017c": "z", "\u1e92": "Z", "\u1e93": "z", "\u1e94": "Z", "\u1e95": "z", "\u01b5": "Z", "\u01b6": "z", "\u0224": "Z", "\u0225": "z", "\u2c6b": "Z", "\u2c6c": "z", "\u1d76": "Z", "\u1d8e": "z", "\u0290": "Z", "\u0291": "z", "\u0240": "Z", "\u1d22": "z", "\uff3a": "Z", "\uff5a": "z"}

    test_df = read_competision_file(False)

    test_text = test_df['comment_text'].astype(str) 

    tokenizer = TokenVectorizer(vector_dict=vocab_emb, lemma_dict=lemma_simple, spell_collector={})

    all_sent, crawl_matrix = tokenizer(test_text, -1)

    for i in range(len(all_sent)):

        all_sent[i] = list(map(lambda x:crawl_matrix[x][0], all_sent[i]))

    return all_sent

"""
import torch

def get_bert_model():

    device = torch.device('cuda')

    bert_config = BertConfig('../input/bert-inference/bert/bert_config.json')

    model = BertForSequenceClassification(bert_config, num_labels=1)

    model.load_state_dict(torch.load("../input/bert-inference/bert/bert_pytorch.bin"))

    model.to(device)

    for param in model.parameters():

        param.requires_grad = False

    model.eval()

    return model
from torch.utils import data

def make_bert_submission(output_file):

    device = torch.device('cuda')

    all_sent = get_bert_tokens()

    model = get_bert_model()

    test_preds = np.zeros((len(all_sent)))

    print(test_preds.shape)

    test = torch.utils.data.TensorDataset(torch.tensor(all_sent, dtype=torch.long))

    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False)

    for i, (x_batch,) in enumerate(test_loader):

        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

        sp = i * 128

        ep = sp + len(pred)

        test_preds[sp:ep] = pred[:, 0].detach().cpu().squeeze().numpy()

    predictions = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()

    df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

    df['prediction'] = predictions

    df.to_csv(output_file, index=False)
"""

Created on Sun Dec 10 11:27:34 2017

@author: Ashish Katiyar

"""

import math

from collections import defaultdict, Iterable

import torch

from copy import deepcopy

from itertools import chain

from torch.autograd import Variable

required = object()

class Optimizer(object):

    """Base class for all optimizers.

    Arguments:

        params (iterable): an iterable of :class:`Variable` s or

            :class:`dict` s. Specifies what Variables should be optimized.

        defaults: (dict): a dict containing default values of optimization

            options (used when a parameter group doesn't specify them).

    """

    def __init__(self, params, defaults):

        self.defaults = defaults

        if isinstance(params, Variable) or torch.is_tensor(params):

            raise TypeError("params argument given to the optimizer should be "

                            "an iterable of Variables or dicts, but got " +

                            torch.typename(params))

        self.state = defaultdict(dict)

        self.param_groups = []

        param_groups = list(params)

        if len(param_groups) == 0:

            raise ValueError("optimizer got an empty parameter list")

        if not isinstance(param_groups[0], dict):

            param_groups = [{'params': param_groups}]

        for param_group in param_groups:

            self.add_param_group(param_group)

    def __getstate__(self):

        return {

            'state': self.state,

            'param_groups': self.param_groups,

        }

    def __setstate__(self, state):

        self.__dict__.update(state)

    def state_dict(self):

        """Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content

            differs between optimizer classes.

        * param_groups - a dict containing all parameter groups

        """

        # Save ids instead of Variables

        def pack_group(group):

            packed = {k: v for k, v in group.items() if k != 'params'}

            packed['params'] = [id(p) for p in group['params']]

            return packed

        param_groups = [pack_group(g) for g in self.param_groups]

        # Remap state to use ids as keys

        packed_state = {(id(k) if isinstance(k, Variable) else k): v

                        for k, v in self.state.items()}

        return {

            'state': packed_state,

            'param_groups': param_groups,

        }

    def load_state_dict(self, state_dict):

        """Loads the optimizer state.

        Arguments:

            state_dict (dict): optimizer state. Should be an object returned

                from a call to :meth:`state_dict`.

        """

        # deepcopy, to be consistent with module API

        state_dict = deepcopy(state_dict)

        # Validate the state_dict

        groups = self.param_groups

        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):

            raise ValueError("loaded state dict has a different number of "

                             "parameter groups")

        param_lens = (len(g['params']) for g in groups)

        saved_lens = (len(g['params']) for g in saved_groups)

        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):

            raise ValueError("loaded state dict contains a parameter group "

                             "that doesn't match the size of optimizer's group")

        # Update the state

        id_map = {old_id: p for old_id, p in

                  zip(chain(*(g['params'] for g in saved_groups)),

                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):

            """Make a deep copy of value, casting all tensors to device of param."""

            if torch.is_tensor(value):

                # Floating-point types are a bit special here. They are the only ones

                # that are assumed to always match the type of params.

                if any(tp in type(param.data).__name__ for tp in {'Half', 'Float', 'Double'}):

                    value = value.type_as(param.data)

                value = value.cuda(param.get_device()) if param.is_cuda else value.cpu()

                return value

            elif isinstance(value, dict):

                return {k: cast(param, v) for k, v in value.items()}

            elif isinstance(value, Iterable):

                return type(value)(cast(param, v) for v in value)

            else:

                return value

        # Copy state assigned to params (and cast tensors to appropriate types).

        # State that is not assigned to params is copied as is (needed for

        # backward compatibility).

        state = defaultdict(dict)

        for k, v in state_dict['state'].items():

            if k in id_map:

                param = id_map[k]

                state[param] = cast(param, v)

            else:

                state[k] = v

        # Update parameter groups, setting their 'params' value

        def update_group(group, new_group):

            new_group['params'] = group['params']

            return new_group

        param_groups = [

            update_group(g, ng) for g, ng in zip(groups, saved_groups)]

        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):

        """Clears the gradients of all optimized :class:`Variable` s."""

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is not None:

                    data = p.grad.data

                    p.grad = Variable(data.new().resize_as_(data).zero_())

    def step(self, closure):

        """Performs a single optimization step (parameter update).

        Arguments:

            closure (callable): A closure that reevaluates the model and

                returns the loss. Optional for most optimizers.

        """

        raise NotImplementedError

    def add_param_group(self, param_group):

        """Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made

        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:

            param_group (dict): Specifies what Variables should be optimized along with group

            specific optimization options.

        """

        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']

        if isinstance(params, Variable):

            param_group['params'] = [params]

        else:

            param_group['params'] = list(params)

        for param in param_group['params']:

            if not isinstance(param, Variable):

                raise TypeError("optimizer can only optimize Variables, "

                                "but one of the params is " + torch.typename(param))

            if not param.requires_grad:

                raise ValueError("optimizing a parameter that doesn't require gradients")

            if not param.is_leaf:

                raise ValueError("can't optimize a non-leaf Variable")

        for name, default in self.defaults.items():

            if default is required and name not in param_group:

                raise ValueError("parameter group didn't specify a value of required optimization parameter " +

                                 name)

            else:

                param_group.setdefault(name, default)

        param_set = set()

        for group in self.param_groups:

            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):

            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

class Nadam(Optimizer):

    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:

        params (iterable): iterable of parameters to optimize or dicts defining

            parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing

            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve

            numerical stability (default: 1e-8)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:

        https://arxiv.org/abs/1412.6980

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay)

        super(Nadam, self).__init__(params, defaults)

    def step(self, lr=None, closure=None):

        """Performs a single optimization step.

        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            loss = closure()

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['prod_mu_t'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:

                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # prod_mu_t = 1

                mu_t = beta1*(1 - 0.5*0.96**(state['step']/250))

                mu_t_1 = beta1*(1 - 0.5*0.96**((state['step']+1)/250))

                prod_mu_t = state['prod_mu_t'] * mu_t

                prod_mu_t_1 = prod_mu_t * mu_t_1

                state['prod_mu_t'] = prod_mu_t

                # for i in range(state['step']):

                #     mu_t = beta1*(1 - 0.5*0.96**(i/250))

                #     mu_t_1 = beta1*(1 - 0.5*0.96**((i+1)/250))

                #     prod_mu_t = prod_mu_t * mu_t

                #     prod_mu_t_1 = prod_mu_t * mu_t_1

                g_hat = grad/(1-prod_mu_t)

                m_hat = exp_avg / (1-prod_mu_t_1)

                m_bar = (1-mu_t)*g_hat + mu_t_1*m_hat

                exp_avg_sq_hat = exp_avg_sq/(1 - beta2 ** state['step'])

                denom = exp_avg_sq_hat.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                #step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                #step_size = group['lr']

                if lr is not None:

                    step_size = lr

                else:

                    step_size = group['lr']

                p.data.addcdiv_(-step_size, m_bar, denom)

                #p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

import numpy as np

import os

import time

import gc

import random

from time import time

from tqdm import tqdm

from fastcache import clru_cache as lru_cache

from textblob import TextBlob



import torch

from torch import nn

from torch.utils import data

from torch.nn import functional as F



time_s = time()

def print_(*args):

    print(time()-time_s, end="\t")

    return print(*args)
def make_isolated_embedding(matrix):

    p_matrix = [0.009182812785100022,-0.040584789317261394,-0.011846708215597239,-0.03270542198528366,0.005831483538662122,-0.0004415112849428413,-0.05269573502126941,0.008208912498716938,0.019490382223248188,-0.02458577452404483,0.028250662928277572,0.18361226878326614,0.021729090546093142,-0.038275015170959006,0.0033973759346653093,0.008467473701057892,-0.028189553307668806,0.024824375401442614,0.0036222467359421444,0.024460037156900954,-0.02378269272174318,-0.01817182542976324,-0.019882241470596283,-0.031097727239060385,-0.00599710407291916,-0.03152349409585279,0.009122488615571177,-0.0001713773609040107,-0.01644240735678459,0.03454689363198803,0.005078502134341189,-0.004377185795367971,0.0048557909975371835,-0.08006686829112608,-0.003760243833561533,-0.0248445653734306,-0.017009404930426887,-0.00569534237879755,-0.000642721609025357,0.03659293694138439,-0.009366632345025928,0.043001355613987345,0.014811545442659625,-0.0553467682511432,0.032799926786718746,-0.03219448863992589,0.03322567142992652,-0.026552534013046455,0.007767168305946927,0.02143737139128056,-0.012692011632648841,-0.06745580507468553,0.2515678896580645,-0.0013208436697328095,-3.301024508568336e-05,-0.0003015832384097134,-0.019003431418042088,-0.00188837459088892,0.00436508684388134,0.011050059997033708,-0.001197451984162603,0.009456930867826065,-0.007320853927069258,-0.09904656272567001,-0.0007289314432978023,0.010478639940070549,-0.00046170855686492897,-0.0018897862799705932,-0.014744774384683417,-0.004896648364511924,-0.04493761518876994,-0.010619764907179256,-0.008544664017908073,0.1375873417196435,-0.011918515484566967,-0.019944713637476748,-0.44255750737304644,-0.02587336684423643,0.09539878700496177,-0.0008147322754549869,0.43128423586384984,-0.0024291757497341086,0.007918524104226929,0.005201713692837981,-0.028053819611239265,-0.012091343156606765,-0.000799473765963778,0.009598349126455482,-0.02050136027628646,0.0032380949567738524,0.0018364292346734206,0.22967682721094831,0.011433104003041313,0.013457276251713958,-0.019701611001235957,-0.0321900493476438,0.01349966601716265,-0.018782597335088132,-0.017765461514827142,-0.023883732314340853,-0.008347606023424563,-0.012389906501522161,0.047805364109126385,-0.008904211179667242,0.005456968065525838,-0.013100424607604846,-0.007764529619285597,-0.008379769811379586,-0.0006800349414373718,0.035364785075270515,0.04445204126762924,-0.026929967458287473,-0.00202610635592915,0.03365310627309023,-0.43408399394956093,-0.022089394180255996,0.013561671001893557,0.010285446738670448,0.01582157729764278,-0.018236547325153906,0.00862505152209495,0.014860190596372995,0.02103445593325409,-0.013663182530340216,0.006236041163585062,-0.031506081879061607,0.02029873972278035,0.007525412512284544,0.2168811591908896,-0.029069495233385707,0.002299838727800742,-0.0061253794425330755,0.016092624984054584,0.004648741671515148,0.03265056138237865,-0.020335595885716887,-0.04517213831750648,-0.016729069807001527,0.02046132553829288,-0.22529141041734044,0.00387213584272845,-0.04590042121987848,0.010424768741486743,0.15805529411390468,-0.003857944676256511,-0.012855499964437788,-0.0031479227685063827,0.01376401982408701,-0.0008840460016189583,0.009722078630268836,-0.0039317025854035195,-0.019016283723644656,0.018775732975998105,-0.017937779275020417,0.031579564569935,-0.007669803337988481,-0.01687133659116447,-0.0221729483049514,0.029318401603712657,-0.005841326041278884,0.4990321811786259,0.009309199701765958,-0.02190627186795492,-0.0013635854455670434,-0.002199810216732186,-0.001540567758101668,0.03244679634465029,-0.02243383997032224,0.029107571828807473,-0.0034612887306351334,-0.01922863481025543,0.033475337590504854,0.027698550868682285,-0.018233073046847306,0.029581502701051407,-0.011272991897479179,-0.00635325244461321,-0.005290278157336871,-0.0012130241688268829,-0.01570584513601337,0.05570008294731479,0.02094458604337456,0.0027758341978849194,0.012296298175652344,0.007278864172280131,0.002402990513916204,0.014286256434092448,-0.027527592602248567,-0.027846720662737783,0.020494460118314867,0.009517022621249824,0.006147198375049207,-0.029689002775387656,0.012213014702094396,-0.005832880658189614,0.002928135512058671,0.01593060614654866,-0.009247318717706099,-0.048770400843015524,0.0070362227117428695,-0.03104945691093978,-0.015234546146962798,0.01427756671938701,0.007851132197927001,-0.010332352055422505,0.01787464251743762,-0.005924500741402286,-0.009247275418077844,0.050408742176736604,-0.025433062568983734,0.033073633640458254,0.5594965267559358,0.02002025955553414,0.006537861022725837,0.020348736349184635,-0.05122281582737965,0.008732368733769882,0.007584178345455834,-0.008786773077186328,0.024175882003831195,-0.02915484864102269,0.17608653925668571,-0.0022359680364203247,-0.016674727443089767,0.02473502033309728,0.024462908760058428,0.0021528612143051967,0.019582923925334672,-0.011121453273058109,0.022039536653415445,-0.018339546014538938,-0.008183806335540509,0.003427008496014203,0.022422575067738395,-0.010886776970292125,-0.14427446303533234,-0.01418545773270993,-0.03238430572051303,0.002060891925426355,-0.002544074362113623,-0.012750695242480304,-0.026500782326316006,-0.019240183126669766,0.01914396654898081,-0.06365170442740913,0.0010214308427048664,0.05724444478284752,0.01443026011061149,0.010996966732755688,0.014228829194472964,0.004425919209124025,0.023091573204805673,-0.01407635056188351,0.006131513263332672,0.017010445713140893,0.028046560357275086,0.011752955298156946,-0.03539766241703242,0.018111176525338966,-0.009629944099677398,0.005488088929932913,0.024416553691102957,0.00799905391368688,-0.03976269392045281,-0.01445453848740384,0.008963378241604711,0.0027252719975425737,0.018520746431111717,-0.051008665560630814,-0.0276821147291433,-0.0031133075706395036,-0.013730677829329312,-0.00988280967096081,-0.022789165765726507,-0.013418135885518098,-0.016176950139522125,0.029539708971903212,0.0023987474714475907,0.0007228514612702998,0.001883056702848162,2.8693485735020397e-05,0.022819038719717185,-0.011042038850970244,0.0016225553727346816,0.000765848399744972,-0.015733646468002856,0.0002460813122046364,0.004383365873786323,0.0018234625955432856,0.008816511465642465,-0.004697932903667204,0.017899667797015045,0.016273725769387452,-0.007007963750189115,-0.006003545090912512,-0.051004466413320376,-0.006022282213469877,-0.015093700763854715,-0.019613736605590657,0.02501420948335617]

    n_matrix = [0.002427948938680964,-0.026350857341430108,-0.01523373394668481,-0.028005489746274863,0.004488919278566633,0.0025764452876863887,-0.05505243786917943,0.011080624352782405,0.018293541273728998,-0.01916369048950003,0.011849926040022529,0.1726957534637598,0.02671063208648816,-0.0369202165242552,0.002791046706245471,0.002561140139550723,-0.014765278434723918,0.02653718321829409,0.007576875917334894,0.02357230384341765,-0.0209466817996944,-0.022988040990450615,-0.01977003211455475,-0.020617486769208188,-0.008075158050003514,-0.0179954635366672,0.006806415834599482,0.0011730936942883922,-0.012051379742019114,0.010921753093303508,-0.001154570365234572,-0.008862633087375532,0.0027804213809410126,-0.07781073560502591,-0.004484659507214222,-0.023580608532469592,-0.020276033932850976,0.0014621975587638385,-0.0048490894033213884,0.035544478277694906,-0.010430916969976944,0.040756492271460484,0.012992691111281212,-0.06436971529312906,0.025517813396807757,-0.02138459024632788,0.020928812631391694,-0.022389823873997926,0.008460295061238595,0.04025238390058873,-0.014675050104351021,-0.05888850239540475,0.24121504093944438,-0.009924799986949144,-0.002962292819389109,0.010771495266329164,-0.014973904330258992,-0.010481301415531025,0.00904944982868231,0.014351674629124914,-0.0017139671479696242,0.0022342122900886638,-0.008853865242500552,-0.10524972617760524,-0.012833226056431994,0.006743596493284535,0.0007670770004786059,0.006012935550680465,-0.010122846041464167,0.006084273449751132,-0.04452407145501877,0.0007107327684431172,-0.006312375965486463,0.1182880331765001,-0.016177655874426787,-0.018447948985036645,-0.4424178339546986,-0.018872906542617345,0.08897691119673731,-0.00470415901595502,0.436660468941245,0.0032877133727051045,0.009196369954346289,-0.002068010661233682,-0.025031863949072013,-0.00926069690416934,0.005462458250593397,0.0021068353560220665,-0.01709488146118999,-0.0029566709210736507,0.000943620246203871,0.2317172252976802,0.008806959870494525,0.00752088202903455,-0.01449295517566327,-0.017681022252292778,0.016020909247113465,-0.01145472112074367,-0.010618605714498804,-0.02522631336859578,-0.004938079177010351,-0.008660230542255298,0.054626019504476805,-0.002525091526663276,-0.005733962870373745,-0.005273452676593823,-0.0065782046329970565,-0.00017058415938834653,-0.0027915610957764763,0.03064282316411414,0.03284994307001055,-0.020904405426359192,-0.00020843298911267657,0.029267702853651444,-0.4260006183631928,-0.02237897081887229,0.010642088237609242,0.010763545739179573,0.0262600524725892,-0.011103068722273873,-0.0016078490097023298,0.010304944165445345,0.019162263645207865,-0.02067159837124151,0.0031207210034887908,-0.02843001930552024,0.012618934736768568,0.011433547789040035,0.22596976546798553,-0.03266044659799642,0.0018090003443324589,-0.013437553406251112,0.012028592313833273,0.001580889141141905,0.03961032739943911,-0.021245330451863238,-0.057640725834056314,-0.015443654799910172,0.01206629875735433,-0.21788295041073666,0.004999618692175123,-0.03360491136628174,0.012228892379779128,0.15182361061559915,0.0024251984860948427,-0.01863088311584819,-0.006704304497265216,0.003544175694853773,-0.010097172415480617,0.011831440400283454,0.0015263952300872186,-0.014214391913847595,0.0129691540697708,-0.016836943518607983,0.035253697888307636,-0.013735298400909429,-0.01119266674685763,-0.006467457637194071,0.021054150995139506,-0.00837039441004984,0.4996003039284626,0.01653211540808042,-0.021174870081088575,-0.0032856912452447007,0.0036857646677837894,0.007159742275286415,0.02419119789276778,-0.018161404557629695,0.025356454604348568,0.005035569901911509,-0.015101742715900165,0.016950226810508826,0.016419820357092434,-0.015038229616722236,0.020945601432537524,-0.009955255839281368,-0.0063248732964104265,-0.006259977931110731,-0.009253129006096147,-0.012844042192023921,0.06025393859641254,0.021554418742506565,-0.0022603530022002306,0.008646393059816073,0.0021184950436328813,-0.004328794379672818,0.0042814588130751455,-0.02360843975716069,-0.02270469406733238,0.027494024616281223,0.007354786948958046,0.0011223183696831356,-0.0246659101181,0.001192645231373174,-0.010403856681650388,-0.0008443164548717201,0.022717614738114334,-0.01816314633264307,-0.033544480214339985,0.011123094908872894,-0.021903283900440586,-0.01100938891326716,0.014690020458654723,0.005771770872135895,-0.009854506337303412,0.011973244880088087,-0.010742825727213726,-0.0026397026205206725,0.04712089776265658,-0.018975396543114788,0.03333279000104644,0.558984739978566,0.020200819094475533,0.006675996046620084,0.018603716363358665,-0.030034901499923806,0.005679375227906664,0.00404851041665811,-0.004779768929533408,0.019002550900477273,-0.021261162361613437,0.17052447262529102,0.000436158718672102,-0.00812029326320723,0.014730889369140389,0.01724511392048127,-0.0039528863298191,0.02100576562004079,-0.012447644038238661,0.010956277262089958,-0.02094148069918646,-0.013894715957772925,0.01733663937860758,0.005224431042220273,-0.010317303056153368,-0.13741099185924197,-0.003878403757536332,-0.030590930218256592,-0.008792223722254069,0.0019539696850622936,-0.007479632152379517,-0.029242993362549114,-0.02387135981277437,0.021846376648828582,-0.05826808307071031,0.00019395829280463247,0.057614417204062324,0.015150731664393562,0.00825988878455156,0.017728418725794937,0.009028666742709043,0.003885303805289234,-0.011138437613912942,0.010253049753112668,0.01218080460788482,0.020737996798181318,0.007403862005747379,-0.03330315142330753,0.008202620815249584,-0.008916952426693952,0.0018534995258902161,0.016552841208331094,0.004449248663274668,-0.026860492048011057,0.0015717269145302107,0.0030196489153873993,-0.0003977291228831768,0.022367249416718627,-0.04058160161507501,-0.022791272163030264,-0.005104002063476328,-0.013945965326021218,-0.005952573189171354,-0.03374339136195488,-0.010478052003909786,-0.018357175565627794,0.024262892596812666,-0.010352915520055997,0.004576074605877353,0.008566841547565592,0.0016967421676360955,0.03243168074292958,-0.0070771955049480605,0.005404794325734851,0.0037958903726275322,-0.0068716697568778655,0.0011996641922105858,0.010420590074357748,0.008913655834798362,0.012203324270666085,-0.005871938924270028,0.009803622712649616,0.021118386310681032,-0.010513438340266966,-0.005160788638567107,-0.04590735133845974,-0.0018737496628913825,-0.01246849510818393,-0.010669066550052973,0.028025055654708263]

    p_embedding = matrix - p_matrix

    n_embedding = matrix - n_matrix

    return np.concatenate([p_embedding, n_embedding], axis=-1)
def make_token(n_tmpsize):

    print_('Read file.')

    train = read_competision_file(train=True)

    test = read_competision_file(train=False)

    train_size = len(train)

    test_size = len(test)

    all_text = list(train['comment_text']) + list(test['comment_text'])

    del train, test

    gc.collect()

    lemma_simple = {"\u00c1": "A", "\u1d8f": "a", "\u00e1": "a", "\u00c0": "A", "\u00e0": "a", "\u00c2": "A", "\u00e2": "a", "\u01cd": "A", "\u01ce": "a", "\u0102": "A", "\u0103": "a", "\u00c3": "A", "\u00e3": "a", "\u1ea2": "A", "\u1ea3": "a", "\u0226": "A", "\u0227": "a", "\u1ea0": "A", "\u1ea1": "a", "\u00c4": "A", "\u00e4": "a", "\u00c5": "A", "\u00e5": "a", "\u1e00": "A", "\u1e01": "a", "\u0100": "A", "\u0101": "a", "\u0104": "A", "\u0105": "a", "\u023a": "A", "\u2c65": "a", "\u0200": "A", "\u0201": "a", "\u1ea4": "A", "\u1ea5": "a", "\u1ea6": "A", "\u1ea7": "a", "\u1eaa": "A", "\u1eab": "a", "\u1ea8": "A", "\u1ea9": "a", "\u1eac": "A", "\u1ead": "a", "\u1eae": "A", "\u1eaf": "a", "\u1eb0": "A", "\u1eb1": "a", "\u1eb4": "A", "\u1eb5": "a", "\u1eb2": "A", "\u1eb3": "a", "\u1eb6": "A", "\u1eb7": "a", "\u01fa": "A", "\u01fb": "a", "\u01e0": "A", "\u01e1": "a", "\u01de": "A", "\u01df": "a", "\u0202": "A", "\u0203": "a", "\u2c6d": "A", "\u0251": "a", "\uff21": "A", "\uff41": "a", "\u1e02": "B", "\u1e03": "b", "\u1e04": "B", "\u1e05": "b", "\u1e06": "B", "\u1e07": "b", "\u0243": "B", "\u0180": "b", "\u0181": "B", "\u0253": "b", "\u0182": "B", "\u0183": "b", "\u1d6c": "B", "\u1d80": "b", "\u0299": "B", "\uff22": "b", "\uff42": "B", "\u0238": "d", "\u2114": "B", "\u0106": "C", "\u0107": "c", "\u0108": "C", "\u0109": "c", "\u010c": "C", "\u010d": "c", "\u010a": "C", "\u010b": "c", "\u0304": "c", "\u00c7": "C", "\u00e7": "c", "\u1e08": "C", "\u1e09": "c", "\u023b": "C", "\u023c": "c", "\u0187": "C", "\u0188": "c", "\u0255": "C", "\u1d04": "c", "\uff23": "C", "\uff43": "c", "\u010e": "D", "\u010f": "d", "\u1e0a": "D", "\u1e0b": "d", "\u1e10": "D", "\u1e11": "d", "\u1e0c": "D", "\u1e0d": "d", "\u1e12": "D", "\u1e13": "d", "\u1e0e": "D", "\u1e0f": "d", "\u0110": "D", "\u0111": "d", "\u0326": "d", "\u0189": "D", "\u0256": "d", "\u018a": "D", "\u0257": "d", "\u018b": "D", "\u018c": "d", "\u1d6d": "D", "\u1d81": "d", "\u1d91": "D", "\u0221": "d", "\u1d05": "D", "\uff24": "d", "\uff44": "D", "\u00c9": "E", "\u00e9": "e", "\u00c8": "E", "\u00e8": "e", "\u00ca": "E", "\u00ea": "e", "\u1e18": "E", "\u1e19": "e", "\u011a": "E", "\u011b": "e", "\u0114": "E", "\u0115": "e", "\u1ebc": "E", "\u1ebd": "e", "\u1e1a": "E", "\u1e1b": "e", "\u1eba": "E", "\u1ebb": "e", "\u0116": "E", "\u0117": "e", "\u00cb": "E", "\u00eb": "e", "\u0112": "E", "\u0113": "e", "\u0228": "E", "\u0229": "e", "\u0118": "E", "\u0119": "e", "\u1d92": "E", "\u0246": "e", "\u0247": "E", "\u0204": "e", "\u0205": "E", "\u1ebe": "e", "\u1ebf": "E", "\u1ec0": "e", "\u1ec1": "E", "\u1ec4": "e", "\u1ec5": "E", "\u1ec2": "e", "\u1ec3": "E", "\u1e1c": "e", "\u1e1d": "E", "\u1e16": "e", "\u1e17": "E", "\u1e14": "e", "\u1e15": "E", "\u0206": "e", "\u0207": "E", "\u1eb8": "e", "\u1eb9": "E", "\u1ec6": "e", "\u1ec7": "E", "\u2c78": "e", "\u1d07": "E", "\uff25": "e", "\uff45": "E", "\u1e1e": "F", "\u1e1f": "f", "\u0191": "F", "\u0192": "f", "\u1d6e": "F", "\u1d82": "f", "\ua730": "F", "\uff26": "f", "\uff46": "F", "\u01f4": "G", "\u01f5": "g", "\u011e": "G", "\u011f": "g", "\u011c": "G", "\u011d": "g", "\u01e6": "G", "\u01e7": "g", "\u0120": "G", "\u0121": "g", "\u0122": "G", "\u0123": "g", "\u1e20": "G", "\u1e21": "g", "\u01e4": "G", "\u01e5": "g", "\u0193": "G", "\u0260": "g", "\u1d83": "G", "\u0262": "g", "\uff27": "G", "\uff47": "g", "\u0124": "H", "\u0125": "h", "\u021e": "H", "\u021f": "h", "\u1e26": "H", "\u1e27": "h", "\u1e22": "H", "\u1e23": "h", "\u1e28": "H", "\u1e29": "h", "\u1e24": "H", "\u1e25": "h", "\u1e2a": "H", "\u1e2b": "h", "\u1e96": "h", "\u0126": "H", "\u0127": "h", "\u2c67": "H", "\u2c68": "h", "\u029c": "H", "\u0266": "h", "\uff28": "H", "\uff48": "h", "\u00cd": "I", "\u00ed": "i", "\u00cc": "I", "\u00ec": "i", "\u012c": "I", "\u012d": "i", "\u00ce": "I", "\u00ee": "i", "\u01cf": "I", "\u01d0": "i", "\u00cf": "I", "\u00ef": "i", "\u1e2e": "I", "\u1e2f": "i", "\u0128": "I", "\u0129": "i", "\u012e": "I", "\u012f": "i", "\u012a": "I", "\u012b": "i", "\u1ec8": "I", "\u1ec9": "i", "\u0208": "I", "\u0209": "i", "\u020a": "I", "\u020b": "i", "\u1eca": "I", "\u1ecb": "i", "\u1e2c": "I", "\u1e2d": "i", "\u0197": "I", "\u0268": "i", "\u1d7b": "I", "\u0130": "I", "\u1d96": "i", "\u026a": "I", "\u0131": "i", "\uff29": "I", "\uff49": "i", "\u0134": "J", "\u0135": "j", "\u0248": "J", "\u0249": "j", "\u01f0": "j", "\u0237": "J", "\u03f3": "j", "\u029d": "J", "\u025f": "j", "\u0284": "J", "\u1d0a": "j", "\uff2a": "J", "\uff4a": "j", "\u1e30": "K", "\u1e31": "k", "\u01e8": "K", "\u01e9": "k", "\u0136": "K", "\u0137": "k", "\u1e32": "K", "\u1e33": "k", "\u1e34": "K", "\u1e35": "k", "\u0198": "K", "\u0199": "k", "\u2c69": "K", "\u2c6a": "k", "\u1d84": "k", "\ua740": "K", "\ua741": "k", "\u1d0b": "K", "\uff2b": "k", "\uff4b": "K", "\u0139": "L", "\u013a": "l", "\u013d": "L", "\u013e": "l", "\u013b": "L", "\u013c": "l", "\u1e36": "L", "\u1e37": "l", "\u1e38": "L", "\u1e39": "l", "\u1e3c": "L", "\u1e3d": "l", "\u1e3a": "L", "\u1e3b": "l", "\u0141": "L", "\u0142": "l", "\u013f": "L", "\u0140": "l", "\u023d": "L", "\u019a": "l", "\u2c60": "L", "\u2c61": "l", "\u2c62": "L", "\u026b": "l", "\u026c": "L", "\u1d85": "l", "\u026d": "L", "\u0234": "l", "\u029f": "L", "\uff2c": "l", "\uff4c": "L", "\u1e3e": "M", "\u1e3f": "m", "\u1e40": "M", "\u1e41": "m", "\u1e42": "M", "\u1e43": "m", "\u1d6f": "M", "\u1d86": "m", "\u2c6e": "M", "\u0271": "m", "\u1d0d": "M", "\uff2d": "m", "\uff4d": "M", "\u0143": "N", "\u0144": "n", "\u01f8": "N", "\u01f9": "n", "\u0147": "N", "\u0148": "n", "\u00d1": "N", "\u00f1": "n", "\u1e44": "N", "\u1e45": "n", "\u0145": "N", "\u0146": "n", "\u1e46": "N", "\u1e47": "n", "\u1e4a": "N", "\u1e4b": "n", "\u1e48": "N", "\u1e49": "n", "\u019d": "N", "\u0272": "n", "\u0220": "N", "\u019e": "n", "\u1d70": "N", "\u1d87": "n", "\u0274": "N", "\u0273": "n", "\uff2e": "N", "\uff4e": "n", "\u014a": "N", "\u014b": "n", "\u0235": "n", "\u00d3": "O", "\u00f3": "o", "\u00d2": "O", "\u00f2": "o", "\u014e": "O", "\u014f": "o", "\u00d4": "O", "\u00f4": "o", "\u1ed0": "O", "\u1ed1": "o", "\u1ed2": "O", "\u1ed3": "o", "\u1ed6": "O", "\u1ed7": "o", "\u1ed4": "O", "\u1ed5": "o", "\u01d1": "O", "\u01d2": "o", "\u00d6": "O", "\u00f6": "o", "\u022a": "O", "\u022b": "o", "\u0150": "O", "\u0151": "o", "\u00d5": "O", "\u00f5": "o", "\u1e4c": "O", "\u1e4d": "o", "\u1e4e": "O", "\u1e4f": "o", "\u022c": "O", "\u022d": "o", "\u022e": "O", "\u022f": "o", "\u0230": "O", "\u0231": "o", "\u00d8": "O", "\u00f8": "o", "\u01fe": "O", "\u01ff": "o", "\u01ea": "O", "\u01eb": "o", "\u01ec": "O", "\u01ed": "o", "\u014c": "O", "\u014d": "O", "\u1e52": "O", "\u1e53": "o", "\u0302": "o", "\u1e50": "O", "\u1e51": "o", "\u1ece": "O", "\u1ecf": "o", "\u020c": "O", "\u020d": "o", "\u020e": "O", "\u020f": "o", "\u01a0": "O", "\u01a1": "o", "\u1eda": "O", "\u1edb": "o", "\u1edc": "O", "\u1edd": "o", "\u1ee0": "O", "\u1ee1": "o", "\u1ede": "O", "\u1edf": "o", "\u1ee2": "O", "\u1ee3": "o", "\u1ecc": "O", "\u1ecd": "o", "\u1ed8": "O", "\u1ed9": "o", "\u019f": "O", "\u0275": "o", "\u2c7a": "O", "\u1d0f": "o", "\uff2f": "O", "\uff4f": "o", "\u1e54": "P", "\u1e55": "p", "\u1e56": "P", "\u1e57": "p", "\u2c63": "P", "\u1d7d": "p", "\u01a4": "P", "\u01a5": "p", "\u1d71": "P", "\u1d88": "p", "\u1d18": "P", "\uff30": "p", "\uff50": "P", "\u024a": "Q", "\u024b": "q", "\uff31": "Q", "\uff51": "q", "\u0239": "Q", "\u02a0": "q", "\u0154": "R", "\u0155": "r", "\u0158": "R", "\u0159": "r", "\u1e58": "R", "\u1e59": "r", "\u0156": "R", "\u0157": "r", "\u0210": "R", "\u0211": "r", "\u0212": "R", "\u0213": "r", "\u1e5a": "R", "\u1e5b": "r", "\u1e5c": "R", "\u1e5d": "r", "\u1e5e": "R", "\u1e5f": "r", "\u024c": "R", "\u024d": "r", "\u2c64": "R", "\u027d": "r", "\u1d72": "R", "\u1d89": "r", "\u027c": "R", "\u027e": "r", "\u1d73": "R", "\u0280": "r", "\uff32": "R", "\uff52": "r", "\u015a": "S", "\u015b": "s", "\u1e64": "S", "\u1e65": "s", "\u015c": "S", "\u015d": "s", "\u0160": "S", "\u0161": "s", "\u1e66": "S", "\u1e67": "s", "\u1e60": "S", "\u1e61": "s", "\u015e": "S", "\u015f": "s", "\u1e62": "S", "\u1e63": "s", "\u1e68": "S", "\u1e69": "s", "\u0218": "S", "\u0219": "s", "\u1d74": "s", "\u1d8a": "s", "\u0282": "s", "\u023f": "s", "\ua731": "s", "\uff33": "S", "\uff53": "s", "\u0164": "T", "\u0165": "t", "\u1e6a": "T", "\u1e6b": "t", "\u0162": "T", "\u0163": "t", "\u1e6c": "T", "\u1e6d": "t", "\u021a": "T", "\u021b": "t", "\u1e70": "T", "\u1e71": "t", "\u1e6e": "T", "\u1e6f": "t", "\u0166": "T", "\u0167": "t", "\u023e": "T", "\u2c66": "t", "\u01ac": "T", "\u01ad": "t", "\u01ae": "T", "\u0288": "t", "\u1e97": "t", "\u1d75": "T", "\u01ab": "t", "\u0236": "T", "\uff34": "t", "\u1d1b": "T", "\uff54": "T", "\u00da": "U", "\u00fa": "u", "\u00d9": "U", "\u00f9": "u", "\u016c": "U", "\u016d": "u", "\u00db": "U", "\u00fb": "u", "\u01d3": "U", "\u01d4": "u", "\u016e": "U", "\u016f": "u", "\u00dc": "U", "\u00fc": "u", "\u01d7": "U", "\u01d8": "u", "\u01db": "U", "\u01dc": "u", "\u01d9": "U", "\u01da": "u", "\u01d5": "U", "\u01d6": "u", "\u0170": "U", "\u0171": "u", "\u0168": "U", "\u0169": "u", "\u1e78": "U", "\u1e79": "u", "\u0172": "U", "\u0173": "u", "\u016a": "U", "\u016b": "u", "\u1e7a": "U", "\u1e7b": "u", "\u1ee6": "U", "\u1ee7": "u", "\u0214": "U", "\u0215": "u", "\u0216": "U", "\u0217": "u", "\u01af": "U", "\u01b0": "u", "\u1ee8": "U", "\u1ee9": "u", "\u1eea": "U", "\u1eeb": "u", "\u1eee": "U", "\u1eef": "u", "\u1eec": "U", "\u1eed": "u", "\u1ef0": "U", "\u1ef1": "u", "\u1ee4": "U", "\u1ee5": "u", "\u1e72": "U", "\u1e73": "u", "\u1e76": "U", "\u1e77": "u", "\u1e74": "U", "\u1e75": "u", "\u0244": "U", "\u0289": "u", "\u1d7e": "U", "\u1d99": "u", "\u1d1c": "U", "\uff35": "u", "\uff55": "U", "\u1e7c": "V", "\u1e7d": "v", "\u1e7e": "V", "\u1e7f": "v", "\u01b2": "V", "\u028b": "v", "\u1d8c": "v", "\u2c71": "V", "\u2c74": "v", "\uff36": "V", "\u1d20": "v", "\uff56": "v", "\u1e82": "W", "\u1e83": "w", "\u1e80": "W", "\u1e81": "w", "\u0174": "W", "\u0175": "w", "\u1e84": "W", "\u1e85": "w", "\u1e86": "W", "\u1e87": "w", "\u1e88": "W", "\u1e89": "w", "\u1e98": "w", "\u2c72": "W", "\u2c73": "w", "\uff37": "W", "\uff57": "w", "\u1e8c": "X", "\u1e8d": "x", "\u1e8a": "X", "\u1e8b": "x", "\u1d8d": "X", "\uff38": "x", "\uff58": "X", "\u00dd": "Y", "\u00fd": "y", "\u1ef2": "Y", "\u1ef3": "y", "\u0176": "Y", "\u0177": "y", "\u0178": "Y", "\u1e99": "y", "\u00ff": "y", "\u1ef8": "Y", "\u1ef9": "y", "\u1e8e": "Y", "\u1e8f": "y", "\u0232": "Y", "\u0233": "y", "\u1ef6": "Y", "\u1ef7": "y", "\u1ef4": "Y", "\u1ef5": "y", "\u024e": "Y", "\u024f": "y", "\u01b3": "Y", "\u01b4": "y", "\uff39": "Y", "\u028f": "y", "\uff59": "y", "\u0179": "Z", "\u017a": "z", "\u1e90": "Z", "\u1e91": "z", "\u017d": "Z", "\u017e": "z", "\u017b": "Z", "\u017c": "z", "\u1e92": "Z", "\u1e93": "z", "\u1e94": "Z", "\u1e95": "z", "\u01b5": "Z", "\u01b6": "z", "\u0224": "Z", "\u0225": "z", "\u2c6b": "Z", "\u2c6c": "z", "\u1d76": "Z", "\u1d8e": "z", "\u0290": "Z", "\u0291": "z", "\u0240": "Z", "\u1d22": "z", "\uff3a": "Z", "\uff5a": "z"}

    tokenizer = TokenVectorizer(vector_dict='../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl', lemma_dict=lemma_simple, spell_collector={}, num_process=3)

    print_("Preprocess.")

    all_text = tokenizer.preprocess(all_text)

    l = train_size//n_tmpsize

    for n in range(n_tmpsize):

        print_("Tokenize %d/%d."%(n+1,n_tmpsize+1))

        fp = l * n

        ep = fp + l

        if n == n_tmpsize-1:

            ep = train_size

        tokens = tokenizer.tokenize(all_text[fp:ep],maxlen=275,pad_sequence=False)

        with open('temporary_tokens_%d.pickle'%(n+1), mode='wb') as f:

            pickle.dump(tokens, f)

        del tokens

        gc.collect()

    print_("Tokenize %d/%d."%(n_tmpsize+1,n_tmpsize+1))

    tokens = tokenizer.tokenize(all_text[train_size:],maxlen=275,pad_sequence=False)

    with open('temporary_tokens_test.pickle', mode='wb') as f:

        pickle.dump(tokens, f)

    del tokens

    gc.collect()

    print_("Vectorize.")

    return tokenizer.vectorize(), test_size
def make_temp(n_tmpsize):

    crawl_matrix, test_size = make_token(n_tmpsize)

    gc.collect()

    print_('Read file.')

    train = read_competision_file(train=True)

    y_train_o = (train['target'].values>=0.5).astype(np.int)

    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

    embedding_matrix = make_isolated_embedding(crawl_matrix)

    return train, embedding_matrix, y_train_o, y_aux_train, test_size
def get_weight(n_tmpsize):

    train, embedding_matrix, y_train_o, y_aux_train, test_size = make_temp(n_tmpsize)

    gc.collect()

    weight_factor = [0.25,0.0,0.0,0.0,0.96,0.34]

    identity_factor_1 = [0.0,0.0,0.0,0.05,0.0,0.0,-0.05,0.0,0.0]

    identity_factor_2 = [-0.05,0.0,0.05,-0.05,0.0,0.0,0.0,0.0,0.0]

    aux_impact_factor = [1.5,1.5,1.2,1.0,1.2,1.2]

    aux_identity_factor = [0.8,0.75,1.15,1.2]

    identity_columns = [

        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    index_subgroup, index_bpsn, index_bnsp = dict(), dict(), dict()

    for col in identity_columns:

        index_subgroup[col] = (train[col].fillna(0).values>=0.5).astype(bool)

        index_bpsn[col] = ( (( (train['target'].values<0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) ) + ( (( (train['target'].values>=0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values<0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) )

        index_bnsp[col] = ( (( (train['target'].values>=0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) ) + ( (( (train['target'].values<0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values<0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) )

    # Overall

    weights = np.ones((len(train),)) * weight_factor[0]

    # Subgroup

    weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) * weight_factor[1]

    weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +

       (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[2]

    weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +

       (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[3]

    weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +

       (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[4]

    weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +

       (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[5]

    index_id1, index_id2 = dict(), dict()

    for col in identity_columns:

        index_id1[col] = (( (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) + (train['target'].values>=0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool)

        index_id2[col] = (( (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) + (train['target'].values<0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool)

    for col,id1 in zip(identity_columns, identity_factor_1):

        weights[index_id1[col]] += id1

    for col,id2 in zip(identity_columns, identity_factor_2):

        weights[index_id2[col]] += id2

    loss_weight = 1.0 / weights.mean()

    weights_aux = np.ones((len(train),))

    weights_aux[(train['target'].values>=0.5).astype(np.int) + (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[0]

    weights_aux[(train['target'].values>=0.5).astype(np.int) + (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[1]

    weights_aux[(train['target'].values<0.5).astype(np.int) + (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[2]

    weights_aux[(train['target'].values<0.5).astype(np.int) + (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[3]

    y_train = np.vstack([y_train_o,weights,weights_aux]).T

    y_train = np.hstack([y_train, y_aux_train])

    return embedding_matrix, y_train, loss_weight, aux_impact_factor, test_size
class GaussianNoise(nn.Module):

    def __init__(self, stddev):

        super(GaussianNoise, self).__init__()

        self.stddev = stddev

    def forward(self, x):

        if self.training:

            noise = torch.empty_like(x)

            noise.normal_(0, self.stddev)

            return x + noise

        else:

            return x



class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x



class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix):

        super(NeuralNet, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], 128, num_layers=2, bidirectional=True, batch_first=True)

        self.noise = GaussianNoise(0.2)

        self.bn = nn.BatchNorm1d(512, momentum=0.5)

        self.hidden1 = nn.Linear(512, 512)

        self.hidden2 = nn.Linear(512, 512)

        self.linear_out = nn.Linear(512, 7)



    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = self.embedding_dropout(h_embedding)

        h_layer, _ = self.lstm(h_embedding)

        avg_pool = torch.mean(h_layer, 1)

        max_pool, _ = torch.max(h_layer, 1)

        h_conc = torch.cat((avg_pool, max_pool), 1)

        h_conc = self.noise(h_conc)

        h_conc = self.bn(h_conc)

        h_conc_linear1 = F.relu(self.hidden1(h_conc))

        h_conc_linear2 = F.relu(self.hidden2(h_conc_linear1))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        out = self.linear_out(hidden)

        return out
def get_batchs(file, batchsize=256):

    with open(file, mode='rb') as f:

        tokens = pickle.load(f)

    for sp in range(0,len(tokens),batchsize):

        ep = min(len(tokens),sp+batchsize)

        tok = tokens[sp:ep]

        nwd = np.max([len(t) for t in tok])

        batch = []

        for i in range(ep-sp):

            l = len(tok[i])

            batch.append(([0]*(nwd-l))+tok[i])

        yield batch
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
class DataLoader2:

    def __init__(self, X, Y, batch_size):

        self.X = X

        self.Y = torch.tensor(Y).cuda()

        self.dataset = self

        self.batch_size = batch_size

        self.batch_pos = 0

    def __next__(self):

        x = torch.tensor(self.X.__next__()).cuda()

        y = self.Y[self.batch_pos:self.batch_pos+len(x)]

        self.batch_pos += len(x)

        return (x,y)

    def __iter__(self):

        return self

    def __len__(self):

        return len(self.Y) // self.batch_size + ( 1 if len(self.Y)%self.batch_size != 0 else 0 )

class DataBunch2:

    def __init__(self, X, Y, batch_size):

        self.path = '.'

        self.device = torch.device('cuda')

        self.train_dl = DataLoader2(X, Y, batch_size)

        self.empty_val = True
from fastai.train import Learner



def weighted_train(num_models=1, n_tmpsize=8, checkpoint_weights=[1,2,4,8], batch_size=256):

    embedding_matrix, y_train, loss_weight, aux_impact_factor, test_size = get_weight(n_tmpsize)

    gc.collect()

    def custom_loss_aux(data, targets):

        ''' Define custom loss function for weighted BCE on 'target' column '''

        bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1].double(),targets[:,:1])

        bce_loss_aux_1 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,1:2].double(),targets[:,3:4])

        bce_loss_aux_2 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,2:3].double(),targets[:,4:5])

        bce_loss_aux_3 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,3:4].double(),targets[:,5:6])

        bce_loss_aux_4 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,4:5].double(),targets[:,6:7])

        bce_loss_aux_5 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,5:6].double(),targets[:,7:8])

        bce_loss_aux_6 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,6:7].double(),targets[:,8:9])

        return (bce_loss_1 * loss_weight) + (bce_loss_aux_1 * aux_impact_factor[0]) + (bce_loss_aux_2 * aux_impact_factor[1]) + (bce_loss_aux_3 * aux_impact_factor[2]) + (bce_loss_aux_4 * aux_impact_factor[3]) + (bce_loss_aux_5 * aux_impact_factor[4]) + (bce_loss_aux_6 * aux_impact_factor[5])

    checkpoint_predictions = []

    model = NeuralNet(embedding_matrix)

    del embedding_matrix

    gc.collect()

    optimizer = Nadam(filter(lambda p: p.requires_grad, model.parameters()))

    weights = []

    checkpoint_predictions = []

    for n in range(num_models):

        before_epoch_weight = 0

        for global_epoch, epoch_weight in enumerate(checkpoint_weights):

            print_("Model#",n,"epoch#",global_epoch)

            l = len(y_train) // n_tmpsize

            y_start = [l * nt for nt in range(n_tmpsize)]

            y_end = [l * (nt+1) for nt in range(n_tmpsize-1)] + [len(y_train)]

            x_files = ['temporary_tokens_%d.pickle'%(nt+1) for nt in range(n_tmpsize)]

            for global_iter, p in enumerate(np.random.permutation(n_tmpsize)):

                databunch = DataBunch2(X=get_batchs(x_files[p]), Y=y_train[y_start[p]:y_end[p]], batch_size=batch_size)

                learn = Learner(databunch, model, loss_func=custom_loss_aux, opt_func=Nadam)

                learn.fit(1)

                test_preds = np.zeros((test_size, 7))

                batch_pos = 0

                with torch.no_grad():

                    for x_batch in get_batchs('temporary_tokens_test.pickle'):

                        X = torch.tensor(x_batch).cuda()

                        test_preds[batch_pos:batch_pos+len(X)] = sigmoid(model(X).detach().cpu().numpy())

                        batch_pos += len(X)

                checkpoint_predictions.append(test_preds[:,0].flatten())

                weights.append(before_epoch_weight + (epoch_weight - before_epoch_weight) * ((global_iter + 1) / n_tmpsize))

            before_epoch_weight = epoch_weight

    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

    return predictions
def make_lstm_submission(output_file):

    predictions = weighted_train()

    df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

    df['prediction'] = predictions

    df.to_csv(output_file, index=False)
make_bert_submission('bert.csv')

make_lstm_submission('lstm.csv')

df1 = pd.read_csv('bert.csv')

df2 = pd.read_csv('lstm.csv')

df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

df['prediction'] = (df1.prediction + df2.prediction) / 2.

df.to_csv('submission.csv', index=False)