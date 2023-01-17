# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk import everygrams, word_tokenize, RegexpTokenizer
import nltk.corpus as corpus
import itertools as it
import matplotlib.pyplot as plt
from IPython.display import HTML
plt.style.use('ggplot')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/amazon_jobs_dataset.csv')
dataset.head()
if sum(dataset.isnull().sum()) > 0: # There exist some NaN values
    dataset.dropna(inplace=True)
title_counts = dataset['Title'].value_counts()
title_counts.head(10)
HTML(f'There are {len(title_counts[title_counts < 5])} titles that have less than 5 jobs.')
stopwords = corpus.stopwords.words()
stopwords.extend(["years", "experience", "related", "field"]) # More words that clutter our results
regex = r'(C\+\+|\w+)' # Remove punctuation unless it's 'C++'
tokenize = lambda x: [item for item in RegexpTokenizer(regex).tokenize(x) if item not in stopwords]
create_grams = lambda x, minlen=2, maxlen=2: [' '.join(ng) for ng in everygrams(tokenize(x), minlen, maxlen)]
ngrams = lambda col, length=(1,4): pd.Series(list(it.chain(*col.apply(create_grams, args=length)))).value_counts()
ngrams(dataset['Title']).head(10).plot.bar();

ngrams(dataset['BASIC QUALIFICATIONS'], length=(2,4)).head(10).plot.bar();
preffered_qualifications = ngrams(dataset['PREFERRED QUALIFICATIONS'], length=(2,4))
preffered_qualifications.head(10).plot.bar();
preffered_qualifications.head(10)
