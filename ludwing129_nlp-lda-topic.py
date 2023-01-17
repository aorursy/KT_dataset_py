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
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
m = pd.read_csv("../input/tf-word-frequency/m12.csv",encoding='latin1')
a = pd.read_csv("../input/tf-word-frequency/m01.csv",encoding='latin1')
ma = pd.read_csv("../input/tf-word-frequency/m02.csv",encoding='latin1')
import nltk
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
m["review_body"] = m["review_body"].apply(lambda x: tokenizer.tokenize(x))

a["review_body"] = a["review_body"].apply(lambda x: tokenizer.tokenize(x))

ma["review_body"] = ma["review_body"].apply(lambda x: tokenizer.tokenize(x))
from nltk.corpus import stopwords
def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stopwords.words('english')]
    return words

m['review_body'] = m['review_body'].apply(lambda x : remove_stopwords(x))
a['review_body'] = a['review_body'].apply(lambda x : remove_stopwords(x))
ma['review_body'] = ma['review_body'].apply(lambda x : remove_stopwords(x))
mb = pd.DataFrame(m["review_body"])
m3 = np.array(mb)
m3 = m3.tolist()

mc = pd.DataFrame(a["review_body"])
m4 = np.array(mc)
m4 = m4.tolist()

md = pd.DataFrame(ma["review_body"])
m5 = np.array(md)
m5 = m5.tolist()

from gensim import corpora,models
m3_dict=corpora.Dictionary(m3[2])
m3_corpus = [m3_dict.doc2bow(i) for i in m3[2]]
m3_corpus
m3_lda = models.LdaModel(m3_corpus,num_topics=5,id2word = m3_dict)
print('December')
for i in range(5):
    print('topic',i)
    print(m3_lda.print_topic(i))
m4_dict=corpora.Dictionary(m4[3])
m4_corpus = [m4_dict.doc2bow(i) for i in m4[3]]
m4_corpus
m4_lda = models.LdaModel(m4_corpus,num_topics=5,id2word = m4_dict)
print('January')
for i in range(5):
    print('topic',i)
    print(m4_lda.print_topic(i))
m5_dict=corpora.Dictionary(m5[2])
m5_corpus = [m5_dict.doc2bow(i) for i in m5[2]]
m5_corpus
m5_lda = models.LdaModel(m5_corpus,num_topics=5,id2word = m5_dict)
print('Febuary')
for i in range(5):
    print('topic',i)
    print(m4_lda.print_topic(i))