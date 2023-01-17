#importing data to be used

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from textblob import TextBlob

import nltk
# we read the required csv file as GBcomments

GBcomments=pd.read_csv('../input/youtube/GBcomments.csv',error_bad_lines=False)
GBcomments.head()
import string
def punc_remover(t):

    no_punc=[char for char in t if char not in string.punctuation]

    return ''.join(no_punc)
GBcomments['comment_text']=GBcomments['comment_text'].astype(str)

GBcomments['comment_text']=GBcomments['comment_text'].apply(punc_remover)
GBcomments.head()
from nltk.corpus import stopwords
sr= stopwords.words('english')

def remstop(t):

    return [word for word in t.split() if word not in sr]

GBcomments['comment_text']=GBcomments['comment_text'].apply(remstop)
GBcomments.head()
def pol(text):

    a=str(text)

    b=TextBlob(a)

    return b.sentiment.polarity
GBcomments['polarity']=GBcomments.comment_text.apply(pol)
def checker(text):

    if text>0:

        return 1

    elif text<0:

        return -1

    else:

        return 0

GBcomments['polarity']=GBcomments['polarity'].apply(checker)
GBcomments.head()
sns.set_style('whitegrid')

sns.countplot(x='polarity',data=GBcomments)
