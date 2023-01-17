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
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec,KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec,KeyedVectors

model = KeyedVectors.load_word2vec_format('/kaggle/input/gensim-word-vectors/word2vec-google-news-300/word2vec-google-news-300', binary=True)  

model = KeyedVectors.load_word2vec_format('/kaggle/input/gensim-word-vectors/glove-wiki-gigaword-300/glove-wiki-gigaword-300')  

for x in model.most_similar('apple'):
    print(x[0],end=', ')
print()
for x in model.most_similar('dog'):
    print(x[0],end=', ')
print()
for x in model.most_similar('marriage'):
    print(x[0],end=', ')
print()
for x in model.most_similar('football'):
    print(x[0],end=', ')
print()
for x in model.most_similar('cat'):
    print(x[0],end=', ')
print()
for x in model.most_similar('patient'):
    print(x[0],end=', ')
print()
for x in model.most_similar('united'):
    print(x[0],end=', ')
print()
for x in model.most_similar('green'):
    print(x[0],end=', ')
print()