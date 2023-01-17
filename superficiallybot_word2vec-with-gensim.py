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
!pip install gensim
from gensim.models import Word2Vec, KeyedVectors

import pandas as pd

import nltk
df = pd.read_csv('../input/worldnews-on-reddit/reddit_worldnews_start_to_2016-11-22.csv')
df.head(10)
newsTitles = df['title'].values
newsTitles
nltk.download('punkt')
newsVec = [nltk.word_tokenize(title) for title in newsTitles]
newsVec
model = Word2Vec(newsVec, min_count = 1, size = 32)
model.most_similar('man')
vec = model['king'] - model['man'] + model['woman']

model.most_similar([vec])
# You can also load the Google's Word2Vec Pretrained 

#https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
model = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', binary = True, limit = 100000)
vec = model['king'] - model['man'] + model['woman']

print(vec)
model.most_similar([vec])
vec = model['Germany'] - model['Berlin'] + model['Paris']
model.most_similar([vec])

# The most similar to Paris is France
vec = model["Cristiano_Ronaldo"] - model["football"] + model["tennis"]
model.most_similar([vec])

# You can see Nadal is in the list of similars