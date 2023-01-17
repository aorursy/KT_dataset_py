%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 7.5]
import numpy as np
import pandas as pd
import string
from collections import Counter
from Crypto.Cipher import *
from cryptography import *
import hashlib, hmac, secrets, base64
from sympy import primerange
import os
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
level_one = train[train['difficulty']==1].copy()
level_one['ciphertext'] = level_one['ciphertext'].map(lambda x: str(x)[:300])
hist = Counter(' '.join(level_one['ciphertext'].astype(str).values))
cph = pd.DataFrame.from_dict(hist, orient='index').reset_index()
cph.columns = ['ciph', 'ciph_freq']
cph = cph.sort_values(by='ciph_freq', ascending=False).reset_index(drop=True)
cph.plot(kind='bar')
# num chars in level 1 cipher
cph['ciph_freq'].sum()
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='train')
news.keys()
news['data'][0:5]
plain = news['data'][:1420]
plain_char_sample = Counter(' '.join(plain))
pt = pd.DataFrame.from_dict(plain_char_sample, orient='index').reset_index()
pt.columns = ['char', 'freq']
pt = pt.sort_values(by='freq', ascending=False).reset_index(drop=True)
pt.plot(kind='bar')
# num plain chars in news sample, very rough now
pt['freq'].sum()
fq_comp=cph
fq_comp['char']=pt['char']
fq_comp['freq']=pt['freq']
fq_comp.head(10)
fq_comp.plot(kind='bar')
more_plain = news['data'][:10000]
more = Counter(' '.join(more_plain))
mpt = pd.DataFrame.from_dict(more, orient='index').reset_index()
mpt.columns = ['char', 'freq']
mpt = mpt.sort_values(by='freq', ascending=False).reset_index(drop=True)
mpt['freq'].sum()
fq_comp2=cph
fq_comp2['char']=mpt['char']
fq_comp2['freq']=mpt['freq']
fq_comp2['ratio']=fq_comp2['freq']/fq_comp2['ciph_freq']
fq_comp2.head(10)
fq_comp2['ratio'].plot(kind='bar')
fq_comp2[['ciph', 'char', 'ratio']].sort_values(by='ratio', ascending=False)
#chars by frequency from this limited sample
fmap=pd.Series(fq_comp2.char.values, fq_comp2.ciph.values).to_dict()
fmap
















