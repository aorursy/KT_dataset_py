#https://github.com/zalandoresearch/flair

!pip install --upgrade pip

!pip install flair
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

from tqdm import tqdm

from bs4 import BeautifulSoup

import re

import matplotlib.pyplot as plt

from datetime import datetime   

    

PATH = "../input/cityofla/CityofLA/Job Bulletins/"
from flair.data import Sentence

from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner-ontonotes-fast') 
def extract_ner(text, TERM='LAW'):

    

    sentence = Sentence(text)

    tagger.predict(sentence)

    r = sentence.to_dict(tag_type='ner')

    

    some_list = []

    for j in range(len(r['entities'])):

        if (r['entities'][j].get("type")) == TERM:

            some_list.append(r['entities'][j].get('text'))

    return some_list
text = []

for fl in tqdm(list(os.listdir("../input/cityofla/CityofLA/Job Bulletins"))):

    with open(PATH+fl, 'r',encoding='latin-1') as f: 

        text.append(f.read())

data = pd.DataFrame({'text':text})
data['title'] = data.text.apply(lambda x: x.split('\n')[0])

data['title'] = data.title.apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ',x))
data.text = data.text.apply(lambda x: x.replace('\t','').split('\n'))

data.text = data.text.apply(lambda x: ' '.join(x))
data['law'] = data.text.apply(lambda x: extract_ner(x))

data['org'] = data.text.apply(lambda x: extract_ner(x, TERM='ORG'))

data['date'] = data.text.apply(lambda x: extract_ner(x, TERM='DATE'))

data['cardinal'] = data.text.apply(lambda x: extract_ner(x, TERM='CARDINAL'))

data['gpe'] = data.text.apply(lambda x: extract_ner(x, TERM='GPE'))

data['event'] = data.text.apply(lambda x: extract_ner(x, TERM='EVENT'))

data['percent'] = data.text.apply(lambda x: extract_ner(x, TERM='PERCENT'))

data['work_of_art'] = data.text.apply(lambda x: extract_ner(x, TERM='WORK_OF_ART'))
data.head(10)
data.to_csv('vacancy_ner.csv', index=False)