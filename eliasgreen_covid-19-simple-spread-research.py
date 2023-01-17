import numpy as np

import pandas as pd 

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import os
all_sources_metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
all_sources_metadata.head()
spread_data = all_sources_metadata[all_sources_metadata['abstract'].str.contains('geographic spread', na = False)]
spread_data.head(1)
for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset'):

    for filename in filenames:

        if '210a892deb1c61577f6fba58505fd65356ce6636' == filename[:-5]:

            print(filename[:-5])
import json

with open('/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/210a892deb1c61577f6fba58505fd65356ce6636.json') as f:

    first_paper = json.load(f)

    print('keys of document:', first_paper.keys())
len(first_paper['body_text'])
print(first_paper['body_text'][0])
first_full_text = ""

for item in first_paper['body_text']:

    first_full_text += item['text']
import nltk.data



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

fisrt_text_sentences = tokenizer.tokenize(first_full_text)

print('\n-----\n'.join(fisrt_text_sentences))
import re

pattern1 = r"\[.*?\]"

fisrt_text_sentences = tokenizer.tokenize(re.sub(pattern1, '', ' '.join(fisrt_text_sentences)))

print('\n-----\n'.join(fisrt_text_sentences))
def hasNumbers(inputString):

    return any(char.isdigit() for char in inputString)
fisrt_text_sentences = filter(hasNumbers, fisrt_text_sentences)

text = '\n-----\n'.join(fisrt_text_sentences)

print(text)
from collections import *

import re



Counter(re.findall(r"[\w']+", text.lower()))
text_stats = sorted(dict(Counter(re.findall(r"[\w']+", text.lower()))).items(), key=lambda x: x[1], reverse=True)
from matplotlib.pyplot import figure

plt.rcParams.update({'font.size': 4})

figure(num=None, figsize=(24, 6), dpi=80)

ax = plt.bar(*zip(*text_stats), color='red', width=0.2)

ticks = plt.xticks(rotation=90)
topics_indexes = [1, 3, 10, 12, 14, 16, 17, 18, 20, 21, 24, -1]

for index in topics_indexes:

    print(text.split('\n-----\n')[index])

    print('\n-----\n')