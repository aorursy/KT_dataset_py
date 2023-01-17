# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
import jieba
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path = "../input/Womens Clothing E-Commerce Reviews.csv"
df = pd.read_csv(path)
df.describe()
clear_df = df.dropna(subset=["Review Text"])
clear_df.describe()
clear_df['tokens'] = clear_df['Review Text'].apply(
    lambda x: " ".join(
        [t for t in filter(lambda xx: xx and len(xx)>1, re.sub("[^0-9a-z]", " ", x.lower()).split(' '))]
    )
)
clear_df['tokens'] ## check data
from sklearn.feature_extraction.text import CountVectorizer
corpus = clear_df['tokens']
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(corpus)  
word = vectorizer.get_feature_names() ## the whole words in corpus 
from sklearn.feature_extraction.text import TfidfTransformer  
transformer = TfidfTransformer()  
tfidf = transformer.fit_transform(X)  
weight = tfidf.toarray()
weight.shape
min_weight = 0.1
core_words = np.apply_along_axis(lambda vec: " ".join([t[0] for t in filter(lambda x: x[1]>min_weight, [(word[j], ele) for j,ele in enumerate(vec)])]), 1, weight)
core_word_df = pd.DataFrame(core_words, columns=["CoreWords"])
core_word_df.shape
final = clear_df.join(core_word_df)
final[['Title', 'Review Text','tokens','CoreWords']]