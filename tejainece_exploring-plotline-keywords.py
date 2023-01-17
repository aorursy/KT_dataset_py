%matplotlib inline



import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from typing import Dict, Tuple, List
df = pd.read_csv('../input/movie_metadata.csv')

df.head()
a = df['plot_keywords'].copy()

b = a.str.split('|')

#.apply(pd.Series, 1).stack()

#a.index.droplevel(-1)
b
s = {} # type: Dict[str, int]

for row in b:

    if isinstance(row, list):

        for tag in row:

            if tag in s:

                s[tag] += 1

            else:

                s[tag] = 1

        

print(s)
stacked = df['plot_keywords'].copy().str.split('|').apply(pd.Series)

stacked.columns = ['one', 'two', 'three', 'four', 'five']

print(stacked[1])
df['plot_keywords'].copy().str.split('|').apply(pd.Series, 1)