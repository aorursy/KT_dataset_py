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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import csv
# list of text documents

text = pd.read_csv('../input/inputData.csv', sep=',',header=0, encoding='utf-8', index_col = 0)
df_orig = pd.DataFrame(text)
df_orig.head()

df = pd.DataFrame()
df['text'] = df_orig['text']
df
from sklearn.feature_extraction.text import CountVectorizer
train_set = df['text']
count_vectorizer = CountVectorizer(stop_words = 'english', ngram_range=(1,2), strip_accents='unicode', analyzer = 'word',
                             lowercase = True)
train_matrix = count_vectorizer.fit_transform(train_set)
print("Vocabulary:", count_vectorizer.vocabulary_)



train_matrix
test_set = "photoshop work with layers" #"photoshop work with layers" #"intensity of colors in photo" #"retouch images" 
test_set = [test_set]

freq_term_matrix = count_vectorizer.transform(test_set)
print(freq_term_matrix)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

tf_idf_train_matrix = tfidf.fit_transform(train_matrix)
print(tf_idf_train_matrix.todense())


vector = tfidf.transform(freq_term_matrix)
print("IDF:", tfidf.idf_)



vector
print(vector.shape)

from sklearn.metrics.pairwise import cosine_similarity
def find_similar(tfidf_matrix, vector, top_n = 5):
    cosine_similarities = cosine_similarity(vector, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


for index, score in find_similar(tf_idf_train_matrix, vector.reshape(1, -1)):
        print(index)
        print (score, df_orig['url'][index])

45
49
47
34
114




