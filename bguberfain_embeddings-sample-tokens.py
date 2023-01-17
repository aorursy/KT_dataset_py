# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/embeddings"))
import codecs

# Any results you write to the current directory are saved as output.
def sample_tokens(fname, n=200, encoding='utf-8'):
    with codecs.open(fname, 'r', encoding=encoding) as f:
        lines = []

        for i in range(n):
            lines.append(f.readline().split(' ', 2)[0])
    return lines
' | '.join(sample_tokens('../input/embeddings/glove.840B.300d/glove.840B.300d.txt'))
' | '.join(sample_tokens('../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'))
' | '.join(sample_tokens('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'))
from gensim.models import KeyedVectors
google_news = KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True, limit=200)
' | '.join(google_news.vocab.keys())