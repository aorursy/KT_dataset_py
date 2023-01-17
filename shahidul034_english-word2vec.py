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
import nltk



text = "Backgammon is one of the oldest:: known board games."

sentences = nltk.sent_tokenize(text)

for sentence in sentences:

    print(sentence)

    
from nltk.tokenize import RegexpTokenizer

tt=""



sentences2=[]

for x in sentences:

    tokenizer = RegexpTokenizer(r'\w+')

    text2=tokenizer.tokenize(x)

    cnt=1

    for x2 in text2:

        if cnt==1:

            tt+=x2

            cnt=0

        else:

            tt+=" "+x2    

    sentences2.append(tt)

    tt=""
sentences2


from nltk.corpus import stopwords



stop_words = set(stopwords.words("english"))

m=[]

for sentence in sentences2:

    words = nltk.word_tokenize(sentence)

    without_stop_words = [word for word in words if not word in stop_words]

    m.append(without_stop_words)
m
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import wordnet

mm=[]

for x in m:

    m2=[]

    for x2 in x:

        lemmatizer = WordNetLemmatizer()

        x3=lemmatizer.lemmatize(x2, wordnet.VERB)

        m2.append(x3)

    mm.append(m2)
mm
from gensim.models import Word2Vec

# train model

model = Word2Vec(mm,size=10,window=10, min_count=1,workers=13)

# summarize the loaded model

print(model)

# summarize vocabulary

words = list(model.wv.vocab)

print(words)

# access vector for one word

print(model['know'])

# save model

model.save('model.bin')

# load model

new_model = Word2Vec.load('model.bin')

print(new_model)
vector = model.wv['know']
vector
w1=['know']

model.wv.most_similar(positive=w1)
model.wv.most_similar(positive=["game", "one"], negative=["Backgammon"], topn=3)
model.wv.doesnt_match(['oldest', 'board', 'one'])
model.wv.similarity('oldest', 'board')