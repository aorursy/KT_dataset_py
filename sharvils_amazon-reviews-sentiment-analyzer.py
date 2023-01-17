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
import nltk

import numpy as np

from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression

from bs4 import BeautifulSoup
wnl=WordNetLemmatizer()
stopwords=set(w.rstrip() for w in open('/kaggle/input/amazon-review-sentiment-analyzer/stopwords.txt'))
pos_rev=BeautifulSoup(open('/kaggle/input/amazon-review-sentiment-analyzer/positive.reviews').read())

pos_rev=pos_rev.findAll('review_text')
neg_rev=BeautifulSoup(open('/kaggle/input/amazon-review-sentiment-analyzer/negative.reviews').read())

neg_rev=neg_rev.findAll('review_text')
np.random.shuffle(pos_rev)

pos_rev=pos_rev[:len(neg_rev)]
def my_tokenizer(s):

    s=s.lower()

    tokens=nltk.tokenize.word_tokenize(s)

    tokens=[t for t in tokens if len(t)>2]

    tokens=[wnl.lemmatize(t) for t in tokens]

    tokens=[t for t in tokens if t not in stopwords]

    return tokens
pos_tokenized=[]

neg_tokenized=[]
word_index_map={}

current_index=0

for review in pos_rev:

    tokens=my_tokenizer(review.text)

    pos_tokenized.append(tokens)

    for token in tokens:

        if token not in word_index_map:

            word_index_map[token]=current_index

            current_index+=1
for review in neg_rev:

    tokens=my_tokenizer(review.text)

    neg_tokenized.append(tokens)

    for token in tokens:

        if token not in word_index_map:

            word_index_map[token]=current_index

            current_index+=1
def tokens_to_vector(tokens,label):

    x=np.zeros(len(word_index_map)+1)

    for t in tokens:

        i=word_index_map[t]

        x[i]+=1

    x=x/x.sum()

    x[-1]=label

    return x

N=len(pos_tokenized)+len(neg_tokenized)

data=np.zeros((N,len(word_index_map)+1))

i=0

for tokens in pos_tokenized:

    xy=tokens_to_vector(tokens,1)

    data[i,:]=xy

    i+=1

for tokens in neg_tokenized:

    xy=tokens_to_vector(tokens,0)

    data[i,:]=xy

    i+=1
np.random.shuffle(data)

X=data[:,:-1]

y=data[:,-1]
X_train=X[:-100,]

y_train=y[:-100,]

X_test=X[-100:,]

y_test=y[-100:,]

model=LogisticRegression()

model.fit(X_train,y_train)

print('Train set acc:',model.score(X_train,y_train))

print('Test set acc:',model.score(X_test,y_test))
threshold=0.5

for word,index in word_index_map.items():

    weight=model.coef_[0][index]

    if weight>threshold or weight<-threshold:

        print(word,weight)