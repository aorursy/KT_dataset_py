import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

from gensim.models import KeyedVectors

# Any results you write to the current directory are saved as output.
model = KeyedVectors.load_word2vec_format('../input/wiki-news-300d-1M.vec')
model.most_similar(positive=['cricket','usa'],negative=['india'])
model['india']