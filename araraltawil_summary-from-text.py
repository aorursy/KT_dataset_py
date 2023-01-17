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



import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
file1 = open("/kaggle/input/data-testing/data.txt","r")

text=file1.read()

print(text)
stopword=list(STOP_WORDS)

nlp=spacy.load('en_core_web_sm')

doc=nlp(text)

tokens=[token.text for token in doc]

punctuation+='\n'
word_frequencies={}
for word in doc:
    if word.text.lower() not in stopword:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[ word.text]=1
            else:
                word_frequencies[ word.text]+=1


max_freq=max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word]=word_frequencies[word]/max_freq


sentence_token=[sent for sent in doc.sents]
sentence_scores={}
for sent in sentence_token:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent]=word_frequencies[word.text.lower()]
            else:
                 sentence_scores[sent]+=word_frequencies[word.text.lower()]
    
select_length=int(len(sentence_token)*0.2)

from heapq import nlargest
summary=nlargest(select_length,sentence_scores,key=sentence_scores.get)


final_summary=[word.text for word in summary]

summary=' '.join(final_summary)

print(summary)