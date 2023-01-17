# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from gensim.models import Word2Vec

import multiprocessing
review = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

review.tail()
import re

STOPWORDS = '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'

cleanwords = re.compile(STOPWORDS)
raw_comments = review['review'].tolist()
all_comments = []

for comments_temp in raw_comments:

    comments = re.sub(cleanwords,' ',comments_temp)

    all_comments.append(comments)
EMD_DIM = 300

w2v = Word2Vec(all_comments, size=EMD_DIM, window=4, min_count=5,negative=5,iter=20, workers=2)
word_vector = w2v.wv
word_vector = w2v.wv
word_vector['dialogue']