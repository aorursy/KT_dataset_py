import re

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

import string

import nltk

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)



%matplotlib inline
import os

print(os.listdir("../input/twitter-sentiment-analysis-hatred-speech"))

train  = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')

test = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/test.csv')
train.head()
def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt    
train['tweet']=np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")
train.head()

train['tweet'] = train['tweet'].str.replace("[^a-zA-Z#]", " ")
train.head()

train['tweet'] = train['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
train['tweet']=train['tweet'].apply(lambda x:x.split())
train.head()
from nltk.stem.porter import *

s = PorterStemmer()

train['tweet'] = train['tweet'].apply(lambda x: [s.stem(i) for i in x])
train.head()
train['tweet']=train['tweet'].apply(lambda x:' '.join(x))
train.head()
from textblob import TextBlob

l=[] 

for i in train['tweet']:

    t=TextBlob(i)

    l.append(t.sentiment.polarity)

    
train['pop']=l
train.head()