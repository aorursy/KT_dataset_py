!pip freeze > kaggle_image_requirements.txt
#Data Processing

import tensorflow as tf

import numpy as np

import unicodedata

import re

import os





MODE = 'train'

BATCH_SIZE = 50

EMBEDDING_SIZE = 256

LSTM_SIZE = 512

NUM_EPOCHS = 300

NUMBER_OF_DATASET = 1000





def read_dataset(number):



    english_data = []

    with open('../input/jw300entw/jw300.en-tw.en') as file:



        line = file.readline()

        cnt = 1

        while line:

            english_data.append(line.strip())

            line = file.readline()

            cnt += 1





    twi_data = []

    with open('../input/jw300entw/jw300.en-tw.tw') as file:



        # twi=file.read()

        line = file.readline()

        cnt = 1

        while line:

            twi_data.append(line.strip())

            line = file.readline()

            cnt += 1



    return english_data[:number],twi_data[:number]

    # return english_data,twi_data



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

raw_data_twi_in = ['<start> ' + normalize_twi(data) for data in raw_data_twi]

raw_data_twi_out = [normalize_twi(data) + ' <end>' for data in raw_data_twi]
data=[]

for tw in raw_data_twi:

#     print(tw)

    

#     print(tw.split())

    data.append(tw.split())
print(data[10])
from gensim.models import Word2Vec



word2vec = Word2Vec(data, min_count=2)
v1 = word2vec.wv['anwummere']

print(v1)
sim_words = word2vec.wv.most_similar('anwummere')

print(sim_words)