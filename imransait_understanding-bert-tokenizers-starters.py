# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

from pytorch_pretrained_bert import BertTokenizer



# Load pre-trained model tokenizer (vocabulary)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
with open("vocab.txt", 'w') as f:

    for token in tokenizer.vocab.keys():

        print (token)

single_chars = []

for token in tokenizer.vocab.keys():

    if len(token) == 1:

        single_chars.append(token)

    

print (single_chars)
print ("Length of single_chars:", len(single_chars))
single_chars_with_hash = []

for token in tokenizer.vocab.keys():

    if len(token) == 3 and token[0:2] == '##':

        single_chars_with_hash.append(token)

    

print (single_chars_with_hash)
print ("Length of single_chars_with_hash:", len(single_chars_with_hash))
words_in_corpus= []

for token in tokenizer.vocab.keys():

    if len(token) > 2 and token[0:2] != '##':

        words_in_corpus.append(token)

    

print (words_in_corpus)
print ("How many words in the corpus:", len(words_in_corpus))
words_in_corpus_with_subwords= []

for token in tokenizer.vocab.keys():

    if len(token) > 2 and token[0:2] == '##':

        words_in_corpus_with_subwords.append(token)

    

print (words_in_corpus_with_subwords)
print ("How many sub words in the corpus:", len(words_in_corpus_with_subwords))
lentgh_of_words_in_corpus= []

for token in tokenizer.vocab.keys():

    if len(token) > 2 and token[0:2] != '##':

        lentgh_of_words_in_corpus.append(len(token))

    

print (lentgh_of_words_in_corpus)
import matplotlib.pyplot as plt

plt.plot(lentgh_of_words_in_corpus)