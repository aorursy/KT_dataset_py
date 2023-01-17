# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from gensim.models import word2vec, KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity
word_vectors = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)
#picking odd one out

print(word_vectors.wv.doesnt_match("breakfast cereal dinner lunch".split()))
test = pd.read_csv('../input/oddoneout/test/Test.csv').values

sample = pd.read_csv('../input/oddoneout/test/sample_submission.csv').values
import sys
def odd_one(words):

    max_sim = sys.maxsize

    odd = None

    for current in words:

        sim = np.sum(cosine_similarity([word_vectors[current]], word_vectors[words]))

        if sim <= max_sim:

            max_sim = sim

            odd = current

        

    return odd
submission = []

for each in test:

    submission.append(odd_one(each))
test
submission