#Data Processing
import tensorflow as tf
import numpy as np
import unicodedata
import re
import os

MODE = 'train'
NUMBER_OF_DATASET = 600000


def read_dataset(number):

    english_data = []
    with open('../input/data-gen-dataset/jw300.en-tw.en') as file:

        line = file.readline()
        cnt = 1
        while line:
            english_data.append(line.strip())
            line = file.readline()
            cnt += 1


    twi_data = []
    with open('../input/data-gen-dataset/jw300.en-tw.tw') as file:

        # twi=file.read()
        line = file.readline()
        cnt = 1
        while line:
            twi_data.append(line.strip())
            line = file.readline()
            cnt += 1

    return english_data[:number],twi_data[:number]

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_eng(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

def normalize_twi(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.ƆɔɛƐ!?’]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s


raw_data_en,raw_data_twi = read_dataset(NUMBER_OF_DATASET)
raw_data_en = [normalize_eng(data) for data in raw_data_en]
raw_data_twi = [normalize_twi(data) for data in raw_data_twi]
data=[]
for tw in raw_data_twi:

    data.append(tw.split())
print(data[10])
from gensim.models import Word2Vec

word2vec = Word2Vec(data,size=100, window=5, min_count=5, workers=4, sg=0)
v1 = word2vec.wv['anwummere']
print(v1)
sim_words = word2vec.wv.most_similar('anwummere')
print(sim_words)
for sw in sim_words:
    print(sw)
from gensim.models import FastText
model_ted = FastText(data, size=100, window=5, min_count=5, workers=4,sg=1)
v1 = model_ted.wv['anwummere']
print(v1)
ft_sim_words= model_ted.wv.most_similar("anwummere")
print(ft_sim_words)
for sw in ft_sim_words:
    print(sw)
