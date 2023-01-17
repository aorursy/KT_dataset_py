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

import re

# Any results you write to the current directory are saved as output.



from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import gensim
def preprocess(art):

    cleaned = art.lower()

    return cleaned

def word_stop(strr):

    strr = re.sub(r'\W+' , ' ' , strr )

    stop_words = stopwords.words('english')

    listt = word_tokenize(strr)

    return ([ token for token in listt  if token not in stop_words ])
f = open("/kaggle/input/covid19.txt", encoding="utf8", errors='ignore')
article  = f.read()
print(article)
pre = preprocess(article)
print(pre)
cleaned_sent = pre.split('.')
all_articles = []

for i in cleaned_sent :

    all_articles.append(word_stop(i))

    
print(all_articles[5:20])
model = gensim.models.Word2Vec(all_articles , size = 96 , window = 5 , min_count=1, workers=2, sg=1 )
test_ = model.most_similar("wuhan" , topn = 5 )

print( test_ )
test_ = model.most_similar("china" , topn = 5 )

print( test_ )
test_ = model.most_similar("disease" , topn = 5 )

print( test_ )
test_ = model.most_similar("infection" , topn = 5 )

print( test_ )
print(model.doesnt_match(["cancer","virus","play"]))
print(model.doesnt_match(["cancer","virus","play","china"]))
print(model.doesnt_match(["cancer","virus","wuhan" ,"infection"]))