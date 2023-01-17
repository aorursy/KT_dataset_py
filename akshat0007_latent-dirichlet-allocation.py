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
data=pd.read_csv("../input/million-headlines/abcnews-date-text.csv")
data_text=data[:5000][['headline_text']];
data_text['index']=data_text.index
documents=data_text
print(len(documents))
documents.head()
documents.isnull().sum()
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,len(documents)):
    text=re.sub('[^a-zA-z]',' ',documents['headline_text'][i])
    text=text.lower()
    text=text.split()
    text=[PorterStemmer().stem(word) for word in text if not word in stopwords.words('english')]
    text=' '.join(text)
    corpus.append(text)
    

for i in range(0,len(corpus)):
    documents['headline_text'][i]=corpus[i]
documents[["headline_text"]]
processed_docs=documents['headline_text']

processed_docs
import gensim

dictionary=gensim.corpora.Dictionary(processed_docs)
count=0
for k,v in dictionary.iteritems():
    print(k,v)
    count+=1
    if count>20:
        break
dictionary.filter_extremes(no_below=15,no_above=0.1,keep_n=1000)
bow_corpus=[dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[100]
bow_corpus_100=bow_corpus[100]
for i in range(len(bow_corpus_100)):
    print("Word {} (\"{}\") appears {} time.".format(bow_corpus_100[i][0], 
                                                     dictionary[bow_corpus_100[i][0]], 
                                                     bow_corpus_100[i][1]))
from gensim import corpora,models
tfidf=models.TfidfModel(bow_corpus)
corpus_tfidf=tfidf[bow_corpus]
lda_model=gensim.models.LdaMulticore(bow_corpus,
                                    num_topics=10,
                                    id2word=dictionary,
                                    passes=2)
for idx,topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 
                                             num_topics=10, 
                                             id2word = dictionary, 
                                             passes = 2, 
                                             workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print("Topic: {} Word: {}".format(idx, topic))
    print("\n")
processed_docs[4310]
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

