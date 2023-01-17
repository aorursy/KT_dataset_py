# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from gensim.models.keyedvectors import KeyedVectors

import time
st=time.time()

wv= KeyedVectors.load_word2vec_format('../input/cc.bn.300.vec')

et=time.time()

dt=et-st

print('dt=',dt)
sm=wv.similarity(u'আম',u'ফল')

print(sm)
sm=wv.similarity(u'আম',u'ফুল')

print(sm)
ws=wv.most_similar(positive=[u'সুন্দর'], topn = 10)
for w in ws:

    print('w=',w[0])
vector = wv['রাজধানীতে']  # numpy vector of a word

print(len(vector))

print(vector)